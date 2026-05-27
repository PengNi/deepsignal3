from __future__ import absolute_import

import argparse
import time
import os
import sys
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from .models import ModelBiLSTM, modelMTM
from .dataloader import SignalFeaData1s, generate_offsets
from .dataloader import clear_linecache

from .utils.process_utils import str2bool
from .utils.process_utils import random_select_file_rows_s
from .utils.process_utils import count_line_num
from .utils.process_utils import concat_two_files
from .utils.process_utils import select_negsamples_asposkmer

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Manager

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def _init_ddp(local_rank, global_world_size, args):
    torch.cuda.set_device(local_rank)
    global_rank = args.node_rank * args.ngpus_per_node + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=global_world_size,
        rank=global_rank,
    )
    return global_rank


def _sync_stop(global_rank, local_rank, test_accus, threshold=0.95):
    """Synchronise early-stopping decision across all ranks via all_reduce."""
    stop_flag = torch.tensor(0.0, device=f"cuda:{local_rank}")
    if global_rank == 0 and len(test_accus) > 0 and np.mean(test_accus) >= threshold:
        stop_flag += 1.0
    dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
    return stop_flag.item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM worker
# ─────────────────────────────────────────────────────────────────────────────

def train_1time(local_rank, global_world_size, train_file, valid_file,
                valid_lidxs, idx2aclogits, args):
    """DDP worker: train BiLSTM on half the data, score the other half."""
    global_rank = _init_ddp(local_rank, global_world_size, args)
    sys.stderr.write(
        "bilstm_worker-{} [init] local_rank={}, global_rank={}\n".format(
            os.getpid(), local_rank, global_rank))

    # ── dataset ──────────────────────────────────────────────────────────────
    train_linenum = count_line_num(train_file, False)
    train_offsets = generate_offsets(train_file)
    train_dataset = SignalFeaData1s(train_file, train_offsets, train_linenum)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True)
    train_loader  = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ModelBiLSTM(
        args.seq_len, args.signal_len, args.layernum1, args.layernum2,
        args.class_num, args.dropout_rate, args.hid_rnn,
        args.n_vocab, args.n_embed,
        str2bool(args.is_base), str2bool(args.is_signallen),
        False,           # is_trace: not used in denoise
        "both_bilstm",
    )
    dist.barrier(device_ids=[local_rank])
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    weight_rank = torch.tensor([1.0, args.pos_weight], dtype=torch.float).cuda(local_rank)
    criterion   = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = StepLR(optimizer, step_size=2, gamma=0.1)

    # ── train ─────────────────────────────────────────────────────────────────
    total_step = len(train_loader)
    if global_rank == 0:
        sys.stderr.write("bilstm_worker total_step: {}\n".format(total_step))

    model.train()
    for epoch in range(args.epoch_num):
        test_accus = []
        train_loader.sampler.set_epoch(epoch)

        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels, _ = sfeatures

            kmer            = kmer[:, args.bias:args.seq_len + args.bias]
            base_means      = base_means[:, args.bias:args.seq_len + args.bias]
            base_stds       = base_stds[:, args.bias:args.seq_len + args.bias]
            base_signal_lens = base_signal_lens[:, args.bias:args.seq_len + args.bias]
            signals         = signals[:, args.bias:args.seq_len + args.bias, :]

            kmer            = kmer.cuda(local_rank, non_blocking=True)
            base_means      = base_means.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            base_stds       = base_stds.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            base_signal_lens = base_signal_lens.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            signals         = signals.cuda(local_rank, non_blocking=True)
            labels          = labels.cuda(local_rank, non_blocking=True)

            x_mask = torch.isnan(signals)
            signals[x_mask] = 0

            outputs, tlogits = model(kmer, base_means, base_stds, base_signal_lens, signals)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if global_rank == 0 and (i + 1) % args.step_interval == 0:
                _, tpredicted = torch.max(tlogits.data, 1)
                i_accuracy = metrics.accuracy_score(labels.cpu().numpy(), tpredicted.cpu().numpy())
                test_accus.append(i_accuracy)
                sys.stderr.write(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}\n".format(
                        epoch + 1, args.epoch_num, i + 1, total_step,
                        loss.item(), i_accuracy))
                sys.stderr.flush()

        scheduler.step()
        if _sync_stop(global_rank, local_rank, test_accus):
            break

    # ── score valid set ───────────────────────────────────────────────────────
    valid_linenum = count_line_num(valid_file, False)
    valid_offsets = generate_offsets(valid_file)
    valid_dataset = SignalFeaData1s(valid_file, valid_offsets, valid_linenum)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, shuffle=False, drop_last=True)
    valid_loader  = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=True,
        sampler=valid_sampler, drop_last=True)

    total_step = len(valid_loader)
    if global_rank == 0:
        sys.stderr.write("valid total_step: {}\n".format(total_step))

    model.eval()
    vlabels_total, vpredicted_total = [], []

    with torch.no_grad():
        lineidx_cnt = 0
        for vi, vsfeatures in enumerate(valid_loader):
            _, vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals, vlabels, _ = vsfeatures

            vkmer            = vkmer[:, args.bias:args.seq_len + args.bias]
            vbase_means      = vbase_means[:, args.bias:args.seq_len + args.bias]
            vbase_stds       = vbase_stds[:, args.bias:args.seq_len + args.bias]
            vbase_signal_lens = vbase_signal_lens[:, args.bias:args.seq_len + args.bias]
            vsignals         = vsignals[:, args.bias:args.seq_len + args.bias, :]

            vkmer            = vkmer.cuda(local_rank, non_blocking=True)
            vbase_means      = vbase_means.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            vbase_stds       = vbase_stds.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            vbase_signal_lens = vbase_signal_lens.unsqueeze(-1).cuda(local_rank, non_blocking=True).float()
            vsignals         = vsignals.cuda(local_rank, non_blocking=True)
            vlabels          = vlabels.cuda(local_rank, non_blocking=True)

            vx_mask = torch.isnan(vsignals)
            vsignals[vx_mask] = 0

            _, vlogits = model(vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals)

            _, vpredicted = torch.max(vlogits.data, 1)
            vlabels_cpu   = vlabels.cpu()
            vpredicted_cpu = vpredicted.cpu()
            vlogits_cpu   = vlogits.cpu()

            vlabels_total    += vlabels_cpu.tolist()
            vpredicted_total += vpredicted_cpu.tolist()

            for alogit in vlogits_cpu.detach().numpy():
                # With shuffle=False, rank k processes positions [k, k+W, k+2W, ...]
                valid_file_pos = lineidx_cnt * global_world_size + global_rank
                if valid_file_pos < len(valid_lidxs):
                    idx2aclogits[valid_lidxs[valid_file_pos]] = alogit[1]
                lineidx_cnt += 1

            if global_rank == 0 and ((vi + 1) % args.step_interval == 0 or (vi + 1) == total_step):
                i_acc = metrics.accuracy_score(vlabels_cpu.numpy(), vpredicted_cpu.numpy())
                sys.stderr.write("===Test, Step [{}/{}], Acc: {:.4f}\n".format(
                    vi + 1, total_step, i_acc))
                sys.stderr.flush()

    if global_rank == 0:
        v_acc  = metrics.accuracy_score(vlabels_total, vpredicted_total)
        v_prec = metrics.precision_score(vlabels_total, vpredicted_total, zero_division=0)
        v_rec  = metrics.recall_score(vlabels_total, vpredicted_total, zero_division=0)
        sys.stderr.write(
            "===Test Final, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}\n".format(
                v_acc, v_prec, v_rec))

    del model
    clear_linecache()


# ─────────────────────────────────────────────────────────────────────────────
# MTM worker
# ─────────────────────────────────────────────────────────────────────────────

def train_1time_mtm(local_rank, global_world_size, train_file, valid_file,
                    valid_lidxs, idx2aclogits, args):
    """
    DDP worker: train modelMTM (d_static=0) on half the data, score the other half.
    Uses d_static=0 — proximity tag is unreliable for plant CHH/CHG and is not
    needed for denoise-level signal/sequence discrimination.
    """
    global_rank = _init_ddp(local_rank, global_world_size, args)
    sys.stderr.write(
        "mtm_worker-{} [init] local_rank={}, global_rank={}\n".format(
            os.getpid(), local_rank, global_rank))

    # ── dataset ──────────────────────────────────────────────────────────────
    train_linenum = count_line_num(train_file, False)
    train_offsets = generate_offsets(train_file)
    train_dataset = SignalFeaData1s(train_file, train_offsets, train_linenum)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True)
    train_loader  = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = modelMTM(
        num_chn=args.mtm_num_base_features + args.n_embed,
        d_static=0,
        num_cls=args.class_num,
        ratios=args.mtm_ratios,
        d_model=args.mtm_hid_rnn,
        r_hid=args.mtm_r_hid,
        drop=args.dropout_rate,
        norm_first=args.mtm_norm_first,
        down_mode=args.mtm_down_mode,
        vocab_size=args.n_vocab,
        embedding_size=args.n_embed,
        temporal_depth=args.mtm_temporal_depth,
    )
    dist.barrier(device_ids=[local_rank])
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    weight_rank = torch.tensor([1.0, args.pos_weight], dtype=torch.float).cuda(local_rank)
    criterion   = nn.CrossEntropyLoss(weight=weight_rank)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = StepLR(optimizer, step_size=2, gamma=0.1)

    n_embed = args.n_embed

    def _forward(kmer, signals):
        """Build MTM inputs and run forward on local_rank device."""
        B, L, S = signals.shape
        signals_flat = signals.view(B, -1, 1).cuda(local_rank, non_blocking=True)
        kmer_expand  = (kmer.long().unsqueeze(2).expand(-1, -1, S)
                        .reshape(B, -1).cuda(local_rank, non_blocking=True))
        x_static     = torch.zeros(B, 1, device=f"cuda:{local_rank}", dtype=torch.long)
        tpos         = (torch.arange(L * S, device=f"cuda:{local_rank}")
                        .unsqueeze(0).expand(B, -1))
        x_mask       = torch.isnan(signals_flat)
        false_mask   = torch.zeros(*x_mask.shape[:-1], n_embed,
                                   device=f"cuda:{local_rank}", dtype=torch.bool)
        x_mask       = torch.cat([x_mask, false_mask], dim=-1)
        return model(signals_flat, kmer_expand, x_mask, tpos, x_static)

    # ── train ─────────────────────────────────────────────────────────────────
    total_step = len(train_loader)
    if global_rank == 0:
        sys.stderr.write("mtm_worker total_step: {}\n".format(total_step))

    model.train()
    for epoch in range(args.epoch_num):
        test_accus = []
        train_loader.sampler.set_epoch(epoch)

        for i, sfeatures in enumerate(train_loader):
            _, kmer, _, _, _, signals, labels, _ = sfeatures
            labels = labels.cuda(local_rank, non_blocking=True)
            logits = _forward(kmer, signals.float())
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if global_rank == 0 and (i + 1) % args.step_interval == 0:
                predicted  = torch.argmax(logits.detach(), dim=1).cpu()
                i_accuracy = metrics.accuracy_score(labels.cpu().numpy(), predicted.numpy())
                test_accus.append(i_accuracy)
                sys.stderr.write(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}\n".format(
                        epoch + 1, args.epoch_num, i + 1, total_step,
                        loss.item(), i_accuracy))
                sys.stderr.flush()

        scheduler.step()
        if _sync_stop(global_rank, local_rank, test_accus):
            break

    # ── score valid set ───────────────────────────────────────────────────────
    valid_linenum = count_line_num(valid_file, False)
    valid_offsets = generate_offsets(valid_file)
    valid_dataset = SignalFeaData1s(valid_file, valid_offsets, valid_linenum)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, shuffle=False, drop_last=True)
    valid_loader  = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=True,
        sampler=valid_sampler, drop_last=True)

    total_step = len(valid_loader)
    if global_rank == 0:
        sys.stderr.write("valid total_step: {}\n".format(total_step))

    model.eval()
    vlabels_total, vpredicted_total = [], []

    with torch.no_grad():
        lineidx_cnt = 0
        for vi, vsfeatures in enumerate(valid_loader):
            _, vkmer, _, _, _, vsignals, vlabels, _ = vsfeatures
            vlogits    = _forward(vkmer, vsignals.float())
            vprobs     = torch.softmax(vlogits, dim=-1).cpu()
            vpredicted = torch.argmax(vprobs, dim=1)
            vlabels_np = vlabels.numpy()

            vlabels_total    += vlabels_np.tolist()
            vpredicted_total += vpredicted.tolist()

            for prob_row in vprobs.numpy():
                valid_file_pos = lineidx_cnt * global_world_size + global_rank
                if valid_file_pos < len(valid_lidxs):
                    idx2aclogits[valid_lidxs[valid_file_pos]] = prob_row[1]
                lineidx_cnt += 1

            if global_rank == 0 and ((vi + 1) % args.step_interval == 0 or (vi + 1) == total_step):
                i_acc = metrics.accuracy_score(vlabels_np, vpredicted.numpy())
                sys.stderr.write("===Test, Step [{}/{}], Acc: {:.4f}\n".format(
                    vi + 1, total_step, i_acc))
                sys.stderr.flush()

    if global_rank == 0:
        v_acc  = metrics.accuracy_score(vlabels_total, vpredicted_total)
        v_prec = metrics.precision_score(vlabels_total, vpredicted_total, zero_division=0)
        v_rec  = metrics.recall_score(vlabels_total, vpredicted_total, zero_division=0)
        sys.stderr.write(
            "===Test Final, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}\n".format(
                v_acc, v_prec, v_rec))

    del model
    clear_linecache()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-rank training orchestration
# ─────────────────────────────────────────────────────────────────────────────

def train_rounds(train_file, iterstr, args, modeltype_str):
    print("\n##########Train Cross Rank##########")
    total_num = count_line_num(train_file, False)
    half_num  = total_num // 2
    fname, fext = os.path.splitext(train_file)
    idxs2logtis_all = {i: [] for i in range(total_num)}

    global_world_size = args.ngpus_per_node * args.nodes
    is_mtm = (getattr(args, "model_class", "bilstm") == "mtm")
    worker_fn = train_1time_mtm if is_mtm else train_1time

    for i in range(args.rounds):
        print("##########Train Cross Rank, Iter {}, Round {}##########".format(iterstr, i + 1))
        if train_file == args.train_file:
            train_file1 = fname + "." + modeltype_str + ".half1" + fext
            train_file2 = fname + "." + modeltype_str + ".half2" + fext
        else:
            train_file1 = fname + ".half1" + fext
            train_file2 = fname + ".half2" + fext
        lidxs1, lidxs2 = random_select_file_rows_s(
            train_file, train_file1, train_file2, half_num, False)

        print("##########Train Cross Rank, Iter {}, Round {}, part1##########".format(iterstr, i + 1))
        manager = Manager()
        idxs22logits = manager.dict()
        mp.spawn(worker_fn, nprocs=args.ngpus_per_node,
                 args=(global_world_size, train_file1, train_file2, lidxs2, idxs22logits, args))

        print("##########Train Cross Rank, Iter {}, Round {}, part2##########".format(iterstr, i + 1))
        idxs12logits = manager.dict()
        mp.spawn(worker_fn, nprocs=args.ngpus_per_node,
                 args=(global_world_size, train_file2, train_file1, lidxs1, idxs12logits, args))

        for idx in idxs22logits.keys():
            idxs2logtis_all[idx].append(idxs22logits[idx])
        for idx in idxs12logits.keys():
            idxs2logtis_all[idx].append(idxs12logits[idx])

        os.remove(train_file1)
        os.remove(train_file2)

    print("##########Train Cross Rank, finished!##########")
    sys.stdout.flush()
    return idxs2logtis_all


# ─────────────────────────────────────────────────────────────────────────────
# Sample cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_samples(train_file, idx2logits, score_cf, is_filter_fn, ori_train_file, modeltype_str):
    print("\n###### clean the samples ######")
    idx2probs = {
        idx: [np.mean(idx2logits[idx]), np.std(idx2logits[idx])]
        for idx in idx2logits.keys()
    }

    idx2prob_pos, idx2prob_neg = [], []
    with open(train_file, 'r') as rf:
        for linecnt, line in enumerate(rf):
            label = int(line.strip().split("\t")[11])
            entry = (linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1])
            (idx2prob_pos if label == 1 else idx2prob_neg).append(entry)

    print("There are {} positive, {} negative samples in total;".format(
        len(idx2prob_pos), len(idx2prob_neg)))

    pos_hc, neg_hc = set(), set()

    idx2prob_pos = sorted(idx2prob_pos, key=lambda x: x[1], reverse=True)
    for idx2prob in idx2prob_pos:
        if idx2prob[1] >= score_cf:
            pos_hc.add(idx2prob[0])
    if is_filter_fn:
        idx2prob_neg = sorted(idx2prob_neg, key=lambda x: x[1])
        for idx2prob in idx2prob_neg:
            if idx2prob[1] < 1 - score_cf:
                neg_hc.add(idx2prob[0])

    left_ratio  = float(len(pos_hc)) / len(idx2prob_pos) if idx2prob_pos else 0
    left_ratio2 = float(len(neg_hc)) / len(idx2prob_neg) if idx2prob_neg else 0
    print("{} ({}) high quality positive samples left, "
          "{} ({}) high quality negative samples left".format(
              len(pos_hc), round(left_ratio, 6), len(neg_hc), round(left_ratio2, 6)))

    fname, fext = os.path.splitext(train_file)
    pfx = "." + modeltype_str if train_file == ori_train_file else ""
    train_clean_pos_file = fname + pfx + ".pos.cf" + str(score_cf) + fext
    wfp = open(train_clean_pos_file, 'w')

    train_clean_neg_file = None
    if is_filter_fn:
        train_clean_neg_file = fname + pfx + ".neg.cf" + str(score_cf) + fext
        wfn = open(train_clean_neg_file, 'w')

    with open(train_file, 'r') as rf:
        for lidx, line in enumerate(rf):
            if lidx in pos_hc:
                wfp.write(line)
            elif is_filter_fn and lidx in neg_hc:
                wfn.write(line)
    wfp.close()
    if is_filter_fn:
        wfn.close()

    print("###### clean the samples, finished! ######")
    sys.stdout.flush()

    if is_filter_fn:
        return train_clean_pos_file, (left_ratio + left_ratio2) / 2, train_clean_neg_file
    else:
        return train_clean_pos_file, left_ratio, None


def _get_all_negative_samples(train_file, modeltype_str):
    fname, fext = os.path.splitext(train_file)
    train_neg_file = fname + ".neg_all." + modeltype_str + fext
    with open(train_neg_file, "w") as wf:
        with open(train_file, 'r') as rf:
            for line in rf:
                if int(line.strip().split("\t")[11]) == 0:
                    wf.write(line)
    return train_neg_file


def _output_linenumber2probs(wfile, idx2logits):
    with open(wfile, "w") as wf:
        for idx in sorted(idx2logits.keys()):
            wf.write("\t".join([str(idx), str(np.mean(idx2logits[idx]))]) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main denoise loop
# ─────────────────────────────────────────────────────────────────────────────

def denoise(args):
    total_start = time.time()

    train_file    = args.train_file
    modeltype_str = args.model_class
    train_neg_file = _get_all_negative_samples(train_file, modeltype_str)

    for iter_c in range(args.iterations):
        print("\n###### cross rank to clean samples, Iter: {} ######".format(iter_c + 1))
        iterstr = str(iter_c + 1)
        idxs2logtis_all = train_rounds(train_file, iterstr, args, modeltype_str)

        if iter_c == 0 and args.fst_iter_prob:
            _output_linenumber2probs(train_file + ".probs_1stiter.txt", idxs2logtis_all)

        is_filter_fn = str2bool(args.is_filter_fn)
        train_clean_pos_file, left_ratio, train_clean_neg_file = clean_samples(
            train_file, idxs2logtis_all, args.score_cf, is_filter_fn,
            args.train_file, modeltype_str)

        if train_file != args.train_file:
            os.remove(train_file)

        print("\n#####concat denoised file#####")
        pos_num = count_line_num(train_clean_pos_file)
        if pos_num > 0:
            fname, fext = os.path.splitext(train_neg_file)
            train_seled_neg_file = fname + ".r" + str(pos_num) + fext
            if train_clean_neg_file is None:
                select_negsamples_asposkmer(train_clean_pos_file, train_neg_file, train_seled_neg_file)
            else:
                neg_num = count_line_num(train_clean_neg_file)
                if pos_num <= neg_num:
                    select_negsamples_asposkmer(
                        train_clean_pos_file, train_clean_neg_file, train_seled_neg_file)
                    os.remove(train_clean_neg_file)
                else:
                    train_seled_neg_file = train_clean_neg_file

            fname, fext = os.path.splitext(args.train_file)
            suffix = ".denoise_fpnp" if is_filter_fn else ".denoise_fp"
            train_file = fname + "." + modeltype_str + suffix + str(iter_c + 1) + fext
            concat_two_files(train_clean_pos_file, train_seled_neg_file, concated_fp=train_file)
            os.remove(train_seled_neg_file)
        else:
            if train_clean_neg_file is not None:
                os.remove(train_clean_neg_file)
            print("WARNING: denoise removed all samples from train_file!")
        os.remove(train_clean_pos_file)
        print("#####concat denoised file, finished!#####")

        if left_ratio >= args.kept_ratio or pos_num == 0:
            break

    os.remove(train_neg_file)
    total_end = time.time()
    print("###### denoised file for training: {}".format(train_file))
    print("###### training totally costs {:.2f} seconds".format(total_end - total_start))


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


def main():
    parser = argparse.ArgumentParser(
        "pyrameth denoise: cross-rank label cleaning for training data.")
    parser.add_argument('--train_file', type=str, required=True,
                        help="combined pos+neg training TSV, balanced by kmer")
    parser.add_argument('--is_filter_fn', type=str, default="no",
                        help="filter false negatives too, 'yes' or 'no', default no")

    # model selection
    parser.add_argument('--model_class', type=str, default="bilstm",
                        choices=["bilstm", "mtm"],
                        help="denoise model: 'bilstm' (default) or 'mtm'")

    # shared hyper-params
    parser.add_argument('--seq_len',     type=int,   default=21)
    parser.add_argument('--signal_len',  type=int,   default=15)
    parser.add_argument('--bias',        type=int,   default=0,
                        help="kmer slice start offset for BiLSTM, default 0")
    parser.add_argument('--class_num',   type=int,   default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--n_vocab',     type=int,   default=16)
    parser.add_argument('--n_embed',     type=int,   default=4)
    parser.add_argument('--pos_weight',  type=float, default=1.0)
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--epoch_num',   type=int,   default=3)
    parser.add_argument('--step_interval', type=int, default=100)

    # BiLSTM hyper-params
    parser.add_argument('--layernum1',   type=int, default=3)
    parser.add_argument('--layernum2',   type=int, default=1)
    parser.add_argument('--hid_rnn',     type=int, default=256)
    parser.add_argument('--is_base',     type=str, default="yes")
    parser.add_argument('--is_signallen', type=str, default="yes")

    # MTM hyper-params (only used when --model_class mtm)
    parser.add_argument('--mtm_num_base_features', type=int, default=1)
    parser.add_argument('--mtm_hid_rnn',    type=int,   default=128)
    parser.add_argument('--mtm_ratios',     type=int,   nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--mtm_r_hid',      type=int,   default=4)
    parser.add_argument('--mtm_norm_first', type=str2bool, default=True)
    parser.add_argument('--mtm_down_mode',  type=str,   default="concat",
                        choices=["concat", "avg", "max"])
    parser.add_argument('--mtm_temporal_depth', type=int, default=2)

    # denoise control
    parser.add_argument('--iterations',  type=int,   default=10)
    parser.add_argument('--rounds',      type=int,   default=3)
    parser.add_argument('--score_cf',    type=float, default=0.5)
    parser.add_argument('--kept_ratio',  type=float, default=0.99)
    parser.add_argument('--fst_iter_prob', action="store_true", default=False)

    # distributed training
    parser.add_argument('--nodes',         type=int, default=1)
    parser.add_argument('--ngpus_per_node', type=int, default=2)
    parser.add_argument('--dist_url',      type=str, default="tcp://127.0.0.1:12315")
    parser.add_argument('--node_rank',     type=int, default=0)
    parser.add_argument('--dl_num_workers', type=int, default=0)

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()
    display_args(args)
    denoise(args)
    print("[main] costs {:.2f} seconds".format(time.time() - total_start))


if __name__ == '__main__':
    main()
