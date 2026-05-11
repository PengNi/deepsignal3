# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn import metrics
import numpy as np
import argparse
import os
import sys
import time
import re

from .models import (
    ModelBiLSTM,
    AggrAttRNN,
    ReadCalibRNN,
    modelMTM
)
from .dataloader import (
    SignalFeaData1s,
    generate_offsets,
    AggregateDataset,
    ReadCalibDataset,
)
from .dataloader import clear_linecache
from .utils.process_utils import display_args
from .utils.process_utils import str2bool
from .utils.process_utils import count_line_num

from .utils.constants_torch import use_cuda

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import timedelta
from torch.cuda.amp import autocast  # type: ignore[import]
from scipy.stats import pearsonr, spearmanr

# add this export temporarily
# https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'

torch.set_float32_matmul_precision('high')

# https://zhuanlan.zhihu.com/p/350301395
# https://github.com/tczhangzhi/pytorch-distributed/blob/master/multiprocessing_distributed.py
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


# https://github.com/dpoulopoulos/examples/blob/feature-group-shuffle-split/distributed/ranzcr/utils.py
def cleanup():
    dist.destroy_process_group()


# https://github.com/dpoulopoulos/examples/blob/feature-group-shuffle-split/distributed/ranzcr/utils.py
# TODO: only for single node, or multi nodes in shared file system?
def checkpoint(model, gpu, model_save_path):
    """Saves the model in master process and loads it everywhere else.

    Args:
        model: the model to save
        gpu: the device identifier
        model_save_path:
    Returns:
        model: the loaded model
    """
    if gpu == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.module.state_dict(), model_save_path)

    # use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    model.module.load_state_dict(
        torch.load(model_save_path, map_location=map_location))


def train_worker(local_rank, global_world_size, args):
    global_rank = args.node_rank * args.ngpus_per_node + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=global_world_size,
        rank=global_rank,
    )

    # device = torch.device("cuda", local_rank)
    # torch.cuda.set_device(local_rank)

    sys.stderr.write("training_process-{} [init] == local rank: {}, global rank: {} ==\n".format(os.getpid(),
                                                                                                local_rank,
                                                                                                global_rank))
    
    # 1. define network
    if global_rank == 0 or args.epoch_sync:
        model_dir = args.model_dir
        if model_dir != "/":
            model_dir = os.path.abspath(model_dir).rstrip("/")
            if local_rank == 0:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                else:
                    model_regex = re.compile(
                        r"" + args.model_class + "\.b\d+_s\d+_epoch\d+\.ckpt*"
                    )
                    for mfile in os.listdir(model_dir):
                        if model_regex.match(mfile) is not None:
                            os.remove(model_dir + "/" + mfile)
            model_dir += "/"

    model = ModelBiLSTM(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        "both_bilstm",
        #local_rank
    )

    if args.init_model is not None:
        sys.stderr.write("training_process-{} loading pre-trained model: {}\n".format(os.getpid(), args.init_model))
        para_dict = torch.load(args.init_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)
    
    if str2bool(args.use_compile):
        try:
            model = torch.compile(model)
        except:
            raise ImportError('torch.compile does not exist in PyTorch<2.0.')

    dist.barrier()

    model = model.cuda(local_rank)
    # DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    
    # 2. define dataloader
    sys.stderr.write("training_process-{} reading data..\n".format(os.getpid()))
    
    train_linenum = count_line_num(args.train_file, False)
    train_offsets = generate_offsets(args.train_file)
    train_dataset = SignalFeaData1s(args.train_file, train_offsets, train_linenum)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.dl_num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    valid_linenum = count_line_num(args.valid_file, False)
    valid_offsets = generate_offsets(args.valid_file)
    valid_dataset = SignalFeaData1s(args.valid_file, valid_offsets, valid_linenum)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
                                                                    shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.dl_num_workers,
                                               pin_memory=True,
                                               sampler=valid_sampler)
    
    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    weight_rank = weight_rank.cuda(local_rank)
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    if args.lr_scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience)
    else:
        raise ValueError("--lr_scheduler is not right!")
    

    # Train the model
    total_step = len(train_loader)
    sys.stderr.write("training_process-{} total_step: {}\n".format(os.getpid(), total_step))
    curr_best_accuracy = 0
    curr_best_accuracy_loc = 0
    curr_lowest_loss = 10000
    v_accuracy_epoches = []
    model.train()
    for epoch in range(args.max_epoch_num):
        # set train sampler
        train_loader.sampler.set_epoch(epoch)

        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels, _ = (
                sfeatures
            )
            kmer = kmer.cuda(local_rank, non_blocking=True)
            base_means = base_means.cuda(local_rank, non_blocking=True)
            base_stds = base_stds.cuda(local_rank, non_blocking=True)
            base_signal_lens = base_signal_lens.cuda(local_rank, non_blocking=True)
            # base_probs = base_probs.cuda(local_rank, non_blocking=True)
            signals = signals.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            # Forward pass
            outputs, _ = model(
                kmer, base_means, base_stds, base_signal_lens, signals
            )
            loss = criterion(outputs, labels)

            # TODO: reduce loss? - no need
            # TODO: maybe don't need barrier() either
            # dist.barrier()
            # loss = reduce_mean(loss, global_world_size)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            tlosses.append(loss.detach().item())
            if global_rank == 0 and ((i + 1) % args.step_interval == 0 or (i + 1) == total_step):
                time_cost = time.time() - start
                sys.stderr.write("Epoch [{}/{}], Step [{}/{}]; "
                                 "TrainLoss: {:.4f}; Time: {:.2f}s\n".format(epoch + 1,
                                                                             args.max_epoch_num, i + 1,
                                                                             total_step, np.mean(tlosses),
                                                                             time_cost))
                sys.stderr.flush()
                start = time.time()
                tlosses = []

        model.eval()
        with torch.no_grad():
            vlosses, vlabels_total, vpredicted_total = [], [], []
            v_meanloss = 10000
            for vsfeatures in valid_loader:
                (
                    _,
                    vkmer,
                    vbase_means,
                    vbase_stds,
                    vbase_signal_lens,
                    vsignals,
                    vlabels,
                    _vtags,
                ) = vsfeatures

                vkmer = vkmer.cuda(local_rank, non_blocking=True)
                vbase_means = vbase_means.cuda(local_rank, non_blocking=True)
                vbase_stds = vbase_stds.cuda(local_rank, non_blocking=True)
                vbase_signal_lens = vbase_signal_lens.cuda(local_rank, non_blocking=True)
                # vbase_probs = vbase_probs.cuda(local_rank, non_blocking=True)
                vsignals = vsignals.cuda(local_rank, non_blocking=True)
                vlabels = vlabels.cuda(local_rank, non_blocking=True)
                voutputs, vlogits = model(
                    vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals
                )
                vloss = criterion(voutputs, vlabels)

                vloss = reduce_mean(vloss, global_world_size)

                _, vpredicted = torch.max(vlogits.data, 1)

                vlabels = vlabels.cpu()
                vpredicted = vpredicted.cpu()

                vlosses.append(vloss.item())
                vlabels_total += vlabels.tolist()
                vpredicted_total += vpredicted.tolist()

            v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
            v_precision = metrics.precision_score(vlabels_total, vpredicted_total)
            v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
            v_meanloss = np.mean(vlosses)

            if v_accuracy > curr_best_accuracy - 0.0001:
                if global_rank == 0:
                    # model.state_dict() or model.module.state_dict()?
                    torch.save(model.module.state_dict(),
                               model_dir + args.model_class +
                               '.b{}_s{}_epoch{}.ckpt'.format(args.seq_len, args.signal_len, epoch + 1))
                # TODO: dist.barrier()? and read/sync model dict?
                if v_accuracy > curr_best_accuracy:
                    curr_best_accuracy = v_accuracy
                    curr_best_accuracy_loc = epoch + 1

                if len(v_accuracy_epoches) > 0 and v_accuracy > \
                        v_accuracy_epoches[-1]:
                    if global_rank == 0:
                        torch.save(model.module.state_dict(),
                                   model_dir + args.model_class +
                                   '.betterthanlast.b{}_s{}_epoch{}.ckpt'.format(args.seq_len,
                                                                                 args.signal_len,
                                                                                 epoch + 1))
            if v_meanloss < curr_lowest_loss:
                curr_lowest_loss = v_meanloss
                no_best_model = False

            v_accuracy_epoches.append(v_accuracy)

            time_cost = time.time() - start
            if global_rank == 0:
                try:
                    last_lr = scheduler.get_last_lr()[0]
                    sys.stderr.write('Epoch [{}/{}]; LR: {:.4e}; '
                                     'ValidLoss: {:.4f}, '
                                     'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                     'Best_acc: {:.4f}; Time: {:.2f}s\n'
                                     .format(epoch + 1, args.max_epoch_num, last_lr,
                                             v_meanloss, v_accuracy, v_precision, v_recall,
                                             curr_best_accuracy, time_cost))
                except Exception:
                    sys.stderr.write('Epoch [{}/{}]; '
                                    'ValidLoss: {:.4f}, '
                                    'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                    'Best_acc: {:.4f}; Time: {:.2f}s\n'
                                    .format(epoch + 1, args.max_epoch_num,
                                            v_meanloss, v_accuracy, v_precision, v_recall,
                                            curr_best_accuracy, time_cost))

                sys.stderr.flush()
        model.train()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            sys.stderr.write("training_process-{} early stop!\n".format(os.getpid()))
            break

        if args.epoch_sync:
            sync_ckpt = model_dir + args.model_class + \
                        '.epoch_sync_node{}.b{}_epoch{}.ckpt'.format(args.node_rank, args.seq_len, epoch + 1)
            checkpoint(model, local_rank, sync_ckpt)

        if args.lr_scheduler == "ReduceLROnPlateau":
            lr_reduce_metric = v_meanloss
            scheduler.step(lr_reduce_metric)
        else:
            scheduler.step()

    if global_rank == 0:
        sys.stderr.write("best model is in epoch {} (Acc: {})\n".format(curr_best_accuracy_loc,
                                                                        curr_best_accuracy))
    clear_linecache()
    cleanup()

def train_worker_mtm(local_rank, global_world_size, args):
    """
    分布式训练函数 (MTM)，集成 AdamW、Compile 优化，并自动清洗保存的模型权重 Key
    [修改] 已移除 AMP (混合精度) 以防止 NaN
    """
    global_rank = args.node_rank * args.ngpus_per_node + local_rank

    # 1. 初始化分布式环境
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=global_world_size,
        rank=global_rank,
        timeout=timedelta(minutes=30)
    )

    sys.stderr.write(f"训练进程-{os.getpid()} [初始化] == 本地 rank: {local_rank}, 全局 rank: {global_rank} ==\n")

    # 2. 目录清理与初始化 (仅主进程执行)
    if global_rank == 0 or args.epoch_sync:
        model_dir = args.model_dir
        if model_dir != "/":
            model_dir = os.path.abspath(model_dir).rstrip("/")
            if local_rank == 0:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                else:
                    # 清理旧的 checkpoint
                    model_regex = re.compile(
                        r"" + args.model_class + r"\.b\d+_s\d+_p\d+_epoch\d+\.ckpt*"
                    )
                    for mfile in os.listdir(model_dir):
                        if model_regex.match(mfile) is not None:
                            os.remove(os.path.join(model_dir, mfile))
            model_dir += "/"

    # 3. 初始化模型
    num_chn = args.mtm_num_base_features + args.n_embed
    
    # 注意：这里使用了之前优化过的 MTM 结构参数
    # 如果你的 args 中还没有 temporal_depth，请确保在 parser 中添加或给默认值
    temporal_depth = getattr(args, 'mtm_temporal_depth', 2) 

    model = modelMTM(
        num_chn=num_chn,
        d_static=args.mtm_d_static,
        num_cls=args.class_num,
        ratios=args.mtm_ratios,
        d_model=args.mtm_hid_rnn,
        r_hid=args.mtm_r_hid,
        drop=args.dropout_rate,
        norm_first=str2bool(args.mtm_norm_first),
        down_mode=args.mtm_down_mode,
        vocab_size=args.n_vocab, 
        embedding_size=args.n_embed,
        temporal_depth=temporal_depth # 传入层数
    )

    # 4. 移动到 GPU
    torch.cuda.set_device(local_rank)
    model = model.cuda(local_rank)

    # 5. [关键优化] torch.compile
    # 放在 DDP 之前
    if str2bool(args.use_compile):
        try:
            sys.stderr.write(f"训练进程-{os.getpid()} 正在编译模型 (torch.compile)...\n")
            model = torch.compile(model, mode="default")
        except Exception as e:
            sys.stderr.write(f"Warning: torch.compile 失败，回退到 eager 模式。错误: {e}\n")

    # 6. DDP 包装
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 7. 数据加载
    sys.stderr.write(f"训练进程-{os.getpid()} 读取数据..\n")
    
    train_linenum = count_line_num(args.train_file, False)
    train_offsets = generate_offsets(args.train_file)
    train_dataset = SignalFeaData1s(args.train_file, train_offsets, train_linenum)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dl_num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    valid_linenum = count_line_num(args.valid_file, False)
    valid_offsets = generate_offsets(args.valid_file)
    valid_dataset = SignalFeaData1s(args.valid_file, valid_offsets, valid_linenum)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dl_num_workers,
        pin_memory=True,
        sampler=valid_sampler
    )

    # 8. 损失函数与优化器
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float().cuda(local_rank)
    criterion = nn.CrossEntropyLoss(weight=weight_rank)

    # [优化] 使用 AdamW
    all_params = list(model.module.parameters())
    weight_decay = 0.01 # 默认给个合理的 weight_decay
    
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(all_params, lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(all_params, lr=args.lr, weight_decay=weight_decay)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(all_params, lr=args.lr, momentum=0.8, weight_decay=weight_decay)
    elif args.optim_type == "AdamW":
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optim_type: {args.optim_type}")

    # [优化] 使用 CosineAnnealingLR (带 Warmup 效果更好，这里先用 Cosine)
    if args.lr_scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience)
    elif args.lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch_num, eta_min=1e-8)
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")

    # [优化] 初始化 AMP Scaler
    # scaler = GradScaler() # [注释] 不使用 Scaler

    # 9. 训练循环变量
    total_step = len(train_loader)
    sys.stderr.write(f"训练进程-{os.getpid()} 总步数: {total_step}\n")
    
    curr_best_accuracy = 0
    curr_best_accuracy_loc = 0
    curr_lowest_loss = 10000
    v_accuracy_epoches = []

    v_best_score_epoches=[]
    curr_best_score = 0
    curr_best_score_loc = 0
    # 辅助函数：清洗 state_dict 中的 compile 前缀
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 torch.compile 产生的 _orig_mod. 前缀
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    model.train()
    patience = args.patience
    no_improve_count = 0
    seq_len = args.seq_len
    signal_len = args.signal_len

    for epoch in range(args.max_epoch_num):
        train_loader.sampler.set_epoch(epoch)

        tlosses = []
        start = time.time()

        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels, tags = sfeatures

            # 数据截取：以数据中心为基准，左右等长截取 seq_len
            _start = kmer.shape[1] // 2 - args.seq_len // 2
            train_kmer = kmer[:, _start:_start + args.seq_len]
            train_signals = signals[:, _start:_start + args.seq_len, :]

            # 数据上 GPU
            train_kmer = train_kmer.cuda(local_rank, non_blocking=True).long()
            train_signals = train_signals.cuda(local_rank, non_blocking=True).float()
            labels = labels.cuda(local_rank, non_blocking=True).long()
            tags = tags.cuda(local_rank, non_blocking=True).long()

            # 数据 Reshape 与 Mask 构造
            batch_size = train_signals.shape[0]
            
            # (B, 21, S) -> (B, 21*S, 1)
            signals_view = train_signals.view(batch_size, -1, 1)
            kmer_expand = train_kmer.repeat_interleave(signal_len, dim=1)
            
            x_mask = torch.isnan(signals_view)
            false_mask = torch.zeros((*x_mask.shape[:-1], args.n_embed), dtype=torch.bool, device=x_mask.device)
            x_mask = torch.cat([x_mask, false_mask], dim=-1)

            t = torch.arange(seq_len * signal_len, device=signals_view.device).repeat(batch_size, 1)
            x_static = tags.unsqueeze(-1)

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    signals_view, kmer_expand, x_mask, t, x_static,
                )
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            tlosses.append(loss.detach().item())

            # 打印日志
            if global_rank == 0 and ((i + 1) % args.step_interval == 0 or (i + 1) == total_step):
                time_cost = time.time() - start
                sys.stderr.write(
                    f"轮次 [{epoch + 1}/{args.max_epoch_num}]，步数 [{i + 1}/{total_step}]；"
                    f"损失: {np.mean(tlosses):.4f}；时间: {time_cost:.2f}s\n"
                )
                sys.stderr.flush()
                start = time.time()
                tlosses = []

        # ------------------------------------------------------------------
        # 验证阶段
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            vlosses, vlabels_total, vpredicted_total = [], [], []
            for vi, vsfeatures in enumerate(valid_loader):
                _, vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals, vlabels, vtags = vsfeatures
                
                _vstart = vkmer.shape[1] // 2 - args.seq_len // 2
                vtrain_kmer = vkmer[:, _vstart:_vstart + args.seq_len]
                vtrain_signals = vsignals[:, _vstart:_vstart + args.seq_len, :]

                vtrain_kmer = vtrain_kmer.cuda(local_rank, non_blocking=True).long()
                vtrain_signals = vtrain_signals.cuda(local_rank, non_blocking=True).float()
                vlabels = vlabels.cuda(local_rank, non_blocking=True).long()
                vtags = vtags.cuda(local_rank, non_blocking=True).long()

                batch_size = vtrain_signals.shape[0]
                vsignals = vtrain_signals.view(batch_size, -1, 1)
                vkmer = vtrain_kmer.repeat_interleave(signal_len, dim=1)

                vx_mask = torch.isnan(vsignals)
                vfalse_mask = torch.zeros((*vx_mask.shape[:-1], args.n_embed), dtype=torch.bool, device=vx_mask.device)
                vx_mask = torch.cat([vx_mask, vfalse_mask], dim=-1)
                vt = torch.arange(seq_len * signal_len, device=vsignals.device).repeat(batch_size, 1)
                vx_static = vtags.unsqueeze(-1)

                # [优化] 验证也开启 autocast 加速
                with autocast(dtype=torch.bfloat16):
                    voutputs = model(
                        vsignals, vkmer, vx_mask, vt, vx_static,
                    )
                    vloss = criterion(voutputs, vlabels)

                vloss = reduce_mean(vloss, global_world_size) # 聚合多卡 Loss

                _, vpredicted = torch.max(voutputs.data, 1)
                vlabels = vlabels.cpu()
                vpredicted = vpredicted.cpu()

                vlosses.append(vloss.item())
                vlabels_total += vlabels.tolist()
                vpredicted_total += vpredicted.tolist()

            v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
            v_precision = metrics.precision_score(vlabels_total, vpredicted_total, zero_division=0)
            v_recall = metrics.recall_score(vlabels_total, vpredicted_total, zero_division=0)
            v_meanloss = np.mean(vlosses)

            v_f1 = metrics.f1_score(vlabels_total, vpredicted_total, zero_division=0)
            v_best_score = (v_accuracy + v_precision + v_recall) / 3
            # ------------------------------------------------------------------
            # 模型保存 (包含 Key 清洗逻辑)
            # ------------------------------------------------------------------
            if v_best_score > curr_best_score - 0.0001 and global_rank == 0:
                # 获取原始 state_dict (自动处理 DDP module 前缀)
                raw_state_dict = model.module.state_dict()
                # [关键] 清洗 torch.compile 产生的 _orig_mod. 前缀
                clean_dict = clean_state_dict(raw_state_dict)

                save_path = model_dir + args.model_class + f'.b{args.seq_len}_s{args.signal_len}_p{args.offset}_epoch{epoch + 1}.ckpt'
                torch.save(clean_dict, save_path)
                
                if v_best_score > curr_best_score:
                    curr_best_score = v_best_score
                    curr_best_score_loc = epoch + 1
                
                if len(v_best_score_epoches) > 0 and v_best_score > v_best_score_epoches[-1]:
                    better_path = model_dir + args.model_class + f'.betterthanlast.b{args.seq_len}_s{args.signal_len}_p{args.offset}_epoch{epoch + 1}.ckpt'
                    torch.save(clean_dict, better_path)

            v_best_score_epoches.append(v_best_score)          

            # 日志输出
            time_cost = time.time() - start
            if global_rank == 0:
                try:
                    last_lr = scheduler.get_last_lr()[0]
                    sys.stderr.write(
                        f"轮次 [{epoch + 1}/{args.max_epoch_num}]；学习率: {last_lr:.4e};"
                        f"验证损失: {v_meanloss:.4f}, "
                        f"准确率: {v_accuracy:.4f}, 精确率: {v_precision:.4f}, 召回率: {v_recall:.4f}, "
                        f"最佳准确率: {curr_best_score:.4f}；时间: {time_cost:.2f}s\n"
                    )
                except Exception:
                    sys.stderr.write(
                        f"轮次 [{epoch + 1}/{args.max_epoch_num}];"
                        f"验证损失: {v_meanloss:.4f}, "
                        f"准确率: {v_accuracy:.4f}, 精确率: {v_precision:.4f}, 召回率: {v_recall:.4f}, "
                        f"最佳准确率: {curr_best_score:.4f}；时间: {time_cost:.2f}s\n"
                    )
                sys.stderr.flush()

        model.train()

        # Early Stopping
        if v_meanloss < curr_lowest_loss:
            no_improve_count = 0
            curr_lowest_loss = v_meanloss
            sys.stderr.write(f"训练进程-{os.getpid()} 验证损失改进至 {v_meanloss:.4f}\n")
        else:
            no_improve_count += 1
            sys.stderr.write(f"训练进程-{os.getpid()} 验证损失无改进 ({no_improve_count}/{patience})\n")

        if no_improve_count >= patience and epoch >= args.min_epoch_num - 1:
            if global_rank == 0:
                sys.stderr.write("Early Stop Triggered.\n")
            dist.barrier()
            break

        # Epoch Sync (如果有需要)
        if args.epoch_sync:
            sync_ckpt = model_dir + args.model_class + \
                        f'.epoch_sync_node{args.node_rank}.b{args.seq_len}_p{args.offset}_epoch{epoch + 1}.ckpt'
            # 同样使用清洗后的 dict
            if local_rank == 0:
                clean_dict = clean_state_dict(model.module.state_dict())
                torch.save(clean_dict, sync_ckpt)
            dist.barrier() # 等待保存完成
            # 其他进程加载 (此处略去加载逻辑，通常 checkpoint 函数会处理)

        # Scheduler Update
        if args.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(v_meanloss)
        else:
            scheduler.step()

    if global_rank == 0:
        sys.stderr.write(f"最佳模型位于 Epoch {curr_best_score_loc} (Score: {curr_best_score})\n")
    
    clear_linecache()
    cleanup()

def train_worker_aggregate(local_rank, global_world_size, args):
    """site-level 聚合模型专用训练（完全参照 train_worker_mtm 风格优化）"""
    global_rank = args.node_rank * args.ngpus_per_node + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=global_world_size,
        rank=global_rank,
        timeout=timedelta(minutes=30)
    )

    sys.stderr.write(f"training_process-{os.getpid()} [init] == local rank: {local_rank}, global rank: {global_rank} ==\n")

    # ==================== 模型目录清理 ====================
    if global_rank == 0 or args.epoch_sync:
        model_dir = args.model_dir
        if model_dir != "/":
            model_dir = os.path.abspath(model_dir).rstrip("/")
            if local_rank == 0:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                else:
                    model_regex = re.compile(r"aggregate\.(attbigru|transformer)\.b\d+_epoch\d+\.ckpt*")
                    for mfile in os.listdir(model_dir):
                        if model_regex.match(mfile):
                            os.remove(os.path.join(model_dir, mfile))
            model_dir += "/"

    # ==================== 加载模型（支持 AggrAttRNN 或 AggrTransformer） ====================
    model = AggrAttRNN(
        seq_len=11, num_layers=1, num_classes=1, dropout_rate=args.dropout_rate,
        hidden_size=args.aggregate_hidden, binsize=20,
        model_type='attbigru', device=local_rank
    )

    if args.init_model is not None:
        sys.stderr.write(f"training_process-{os.getpid()} loading pre-trained model: {args.init_model}\n")
        para_dict = torch.load(args.init_model, map_location='cpu')
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in para_dict.items() if k in model_dict})
        model.load_state_dict(model_dict)

    # ==================== torch.compile（与 mtm 完全一致） ====================
    if str2bool(args.use_compile):
        try:
            sys.stderr.write(f"training_process-{os.getpid()} 正在编译模型 (torch.compile)...\n")
            model = torch.compile(model, mode="default")
        except Exception as e:
            sys.stderr.write(f"Warning: torch.compile 失败，回退到 eager 模式。错误: {e}\n")

    dist.barrier()
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ==================== 数据加载（与你之前 npz 一致） ====================
    sys.stderr.write(f"training_process-{os.getpid()} reading aggregate data..\n")
    train_dataset = AggregateDataset(args.train_file)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.dl_num_workers, pin_memory=True
    )

    valid_dataset = AggregateDataset(args.valid_file)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.dl_num_workers, pin_memory=True
    )

    # ==================== 优化器 & 调度器（AdamW + Cosine） ====================
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch_num, eta_min=1e-8)

    # ==================== 训练循环（完全对标 mtm 风格） ====================
    total_step = len(train_loader)
    sys.stderr.write(f"training_process-{os.getpid()} total_step: {total_step}\n")

    curr_lowest_loss = float('inf')
    curr_best_epoch = 0
    no_improve_count = 0
    patience = args.patience

    model.train()
    for epoch in range(args.max_epoch_num):
        train_sampler.set_epoch(epoch)
        tlosses = []
        start = time.time()

        for i, (pos, hist, label) in enumerate(train_loader):
            pos = pos.cuda(local_rank, non_blocking=True)
            hist = hist.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(pos.unsqueeze(-1), hist).squeeze(-1)          # (B,1) -> (B,)
            loss = criterion(outputs, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tlosses.append(loss.item())

            if global_rank == 0 and ((i + 1) % args.step_interval == 0 or (i + 1) == total_step):
                time_cost = time.time() - start
                sys.stderr.write(
                    f"Aggregate Epoch [{epoch+1}/{args.max_epoch_num}], Step [{i+1}/{total_step}]; "
                    f"TrainLoss: {np.mean(tlosses):.6f}; Time: {time_cost:.2f}s\n"
                )
                start = time.time()
                tlosses = []

        # ==================== Validation ====================
        model.eval()
        with torch.no_grad():
            vlosses = []
            val_all_preds = []
            val_all_labels = []
            for pos, hist, label in valid_loader:
                pos = pos.cuda(local_rank, non_blocking=True)
                hist = hist.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)

                outputs = model(pos.unsqueeze(-1), hist).squeeze(-1)
                vloss = criterion(outputs, label)
                vloss = reduce_mean(vloss, global_world_size)
                vlosses.append(vloss.item())

                
                if global_rank == 0:
                    preds = torch.sigmoid(outputs)
                    val_all_preds.append(preds.cpu().numpy())
                    val_all_labels.append(label.cpu().numpy())
            
            # gathered_preds = [torch.zeros_like(val_all_preds) for _ in range(global_world_size)]
            # gathered_labels = [torch.zeros_like(val_all_labels) for _ in range(global_world_size)]
            
            # # 汇总到所有进程（或者仅汇总到 rank 0）
            # dist.all_gather(gathered_preds, val_all_preds.cuda(local_rank))
            # dist.all_gather(gathered_labels, val_all_labels.cuda(local_rank))
            v_meanloss = np.mean(vlosses)

            # ==================== Early Stop 计数（所有 rank 同步，避免死锁） ====================
            # v_meanloss 已经过 reduce_mean 聚合，所有 rank 值相同
            # curr_lowest_loss 也在所有 rank 上同步更新，保证比较结果一致
            if v_meanloss < curr_lowest_loss:
                curr_lowest_loss = v_meanloss
                no_improve_count = 0
            else:
                no_improve_count += 1

            # ==================== 保存（仅 rank 0 执行） ====================
            if global_rank == 0:
                val_all_preds = np.concatenate(val_all_preds)
                val_all_labels = np.concatenate(val_all_labels)

                # 计算回归指标
                pcc, _ = pearsonr(val_all_preds, val_all_labels)
                scc, _ = spearmanr(val_all_preds, val_all_labels)
                mae = np.mean(np.abs(val_all_preds - val_all_labels))
                # 清洗 compile 前缀
                state = model.module.state_dict()
                clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

                save_path = model_dir + f"aggregate.{args.aggregate_model_type}.b11_epoch{epoch+1}.ckpt"
                torch.save(clean_state, save_path)

                if no_improve_count == 0:  # 本 epoch 是新最优
                    curr_best_epoch = epoch + 1
                    torch.save(clean_state, model_dir + "aggregate.best.ckpt")

                cur_lr = optimizer.param_groups[0]['lr']
                sys.stderr.write(
                    f"Aggregate Epoch [{epoch+1}/{args.max_epoch_num}]; LR: {cur_lr:.2e}; "
                    f"ValidLoss: {v_meanloss:.6f} | PCC: {pcc:.4f} | SCC: {scc:.4f} | MAE: {mae:.4f}\n"
                    f"BestValLoss: {curr_lowest_loss:.6f} (epoch {curr_best_epoch}) | NoImprove: {no_improve_count}/{patience}\n"
                )

        model.train()

        if no_improve_count >= patience and epoch >= args.min_epoch_num - 1:
            if global_rank == 0:
                sys.stderr.write(f"training_process-{os.getpid()} Early stop at epoch {epoch+1}\n")
            dist.barrier()
            break

        if args.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(v_meanloss)
        else:
            scheduler.step()

    if global_rank == 0:
        sys.stderr.write(f"✅ Best aggregate model at epoch {curr_best_epoch} (ValLoss: {curr_lowest_loss:.6f})\n")

    clear_linecache()
    cleanup()


def train_aggregate_cpu(args):
    """aggregate 模型的单进程 CPU 训练（无 GPU 环境下自动使用）"""
    device = torch.device('cpu')
    sys.stderr.write(f"[aggregate-cpu] No GPU detected, training on CPU.\n")

    model_dir = os.path.abspath(args.model_dir).rstrip("/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        model_regex = re.compile(r"aggregate\.(attbigru|transformer)\.b\d+_epoch\d+\.ckpt*")
        for mfile in os.listdir(model_dir):
            if model_regex.match(mfile):
                os.remove(os.path.join(model_dir, mfile))
    model_dir += "/"

    model = AggrAttRNN(
        seq_len=11, num_layers=1, num_classes=1, dropout_rate=args.dropout_rate,
        hidden_size=args.aggregate_hidden, binsize=20,
        model_type='attbigru', device='cpu'
    )

    if args.init_model is not None:
        sys.stderr.write(f"[aggregate-cpu] loading pre-trained model: {args.init_model}\n")
        para_dict = torch.load(args.init_model, map_location='cpu')
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in para_dict.items() if k in model_dict})
        model.load_state_dict(model_dict)

    model.to(device)

    sys.stderr.write(f"[aggregate-cpu] reading aggregate data..\n")
    train_dataset = AggregateDataset(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.dl_num_workers, pin_memory=False
    )
    valid_dataset = AggregateDataset(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch_num, eta_min=1e-8)

    total_step = len(train_loader)
    sys.stderr.write(f"[aggregate-cpu] total_step: {total_step}\n")

    curr_lowest_loss = float('inf')
    curr_best_epoch = 0
    no_improve_count = 0
    patience = args.patience

    model.train()
    for epoch in range(args.max_epoch_num):
        tlosses = []
        start = time.time()

        for i, (pos, hist, label) in enumerate(train_loader):
            pos, hist, label = pos.to(device), hist.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(pos.unsqueeze(-1), hist).squeeze(-1)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tlosses.append(loss.item())

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                sys.stderr.write(
                    f"Aggregate Epoch [{epoch+1}/{args.max_epoch_num}], Step [{i+1}/{total_step}]; "
                    f"TrainLoss: {np.mean(tlosses):.6f}; Time: {time.time()-start:.2f}s\n"
                )
                start = time.time()
                tlosses = []

        model.eval()
        with torch.no_grad():
            vlosses, val_preds, val_labels = [], [], []
            for pos, hist, label in valid_loader:
                pos, hist, label = pos.to(device), hist.to(device), label.to(device)
                outputs = model(pos.unsqueeze(-1), hist).squeeze(-1)
                vlosses.append(criterion(outputs, label).item())
                val_preds.append(torch.sigmoid(outputs).numpy())
                val_labels.append(label.numpy())

        v_meanloss = np.mean(vlosses)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        pcc, _ = pearsonr(val_preds, val_labels)
        scc, _ = spearmanr(val_preds, val_labels)
        mae = np.mean(np.abs(val_preds - val_labels))

        clean_state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        save_path = model_dir + f"aggregate.{args.aggregate_model_type}.b11_epoch{epoch+1}.ckpt"
        torch.save(clean_state, save_path)

        if v_meanloss < curr_lowest_loss:
            curr_lowest_loss = v_meanloss
            no_improve_count = 0
            curr_best_epoch = epoch + 1
            torch.save(clean_state, model_dir + "aggregate.best.ckpt")
        else:
            no_improve_count += 1

        cur_lr = optimizer.param_groups[0]['lr']
        sys.stderr.write(
            f"Aggregate Epoch [{epoch+1}/{args.max_epoch_num}]; LR: {cur_lr:.2e}; "
            f"ValidLoss: {v_meanloss:.6f} | PCC: {pcc:.4f} | SCC: {scc:.4f} | MAE: {mae:.4f}\n"
            f"BestValLoss: {curr_lowest_loss:.6f} (epoch {curr_best_epoch}) | NoImprove: {no_improve_count}/{patience}\n"
        )

        model.train()

        if no_improve_count >= patience and epoch >= args.min_epoch_num - 1:
            sys.stderr.write(f"[aggregate-cpu] Early stop at epoch {epoch+1}\n")
            break

        if args.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(v_meanloss)
        else:
            scheduler.step()

    sys.stderr.write(f"✅ Best aggregate model at epoch {curr_best_epoch} (ValLoss: {curr_lowest_loss:.6f})\n")
    clear_linecache()


def train_read_calib_cpu(args):
    """Read-level calibration model (ReadCalibRNN) single-process CPU training.

    Training data must be an npz file produced by scripts/prepare_read_calib_data.py,
    containing: window_probs, window_offsets, window_valid, read_feats, labels.
    Loss: binary cross-entropy (BCE) on logit output.
    """
    from scipy.stats import pearsonr

    device = torch.device('cpu')
    sys.stderr.write("[read_calib-cpu] No GPU detected, training on CPU.\n")

    model_dir = os.path.abspath(args.model_dir).rstrip("/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        model_regex = re.compile(r"read_calib\.(attbigru|attbilstm)\.b\d+_epoch\d+\.ckpt*")
        for mfile in os.listdir(model_dir):
            if model_regex.match(mfile):
                os.remove(os.path.join(model_dir, mfile))
    model_dir += "/"

    model = ReadCalibRNN(
        seq_len=args.read_calib_seq_len,
        num_layers=1,
        num_classes=1,
        dropout_rate=args.dropout_rate,
        hidden_size=args.read_calib_hidden,
        n_read_feats=3,
        model_type=args.read_calib_model_type,
    )

    if args.init_model is not None:
        sys.stderr.write(f"[read_calib-cpu] loading pre-trained model: {args.init_model}\n")
        para_dict = torch.load(args.init_model, map_location='cpu')
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in para_dict.items() if k in model_dict})
        model.load_state_dict(model_dict)

    model.to(device)

    sys.stderr.write("[read_calib-cpu] reading ReadCalib data..\n")
    train_dataset = ReadCalibDataset(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.dl_num_workers, pin_memory=False,
    )
    valid_dataset = ReadCalibDataset(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.dl_num_workers, pin_memory=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epoch_num, eta_min=1e-8)

    total_step = len(train_loader)
    sys.stderr.write(f"[read_calib-cpu] total_step: {total_step}\n")

    curr_lowest_loss = float('inf')
    curr_best_epoch = 0
    no_improve_count = 0
    patience = args.patience

    model.train()
    for epoch in range(args.max_epoch_num):
        tlosses = []
        start = time.time()

        for i, (wprobs, woffs, wvalid, rfeats, labels) in enumerate(train_loader):
            wprobs   = wprobs.to(device)
            woffs    = woffs.to(device)
            wvalid   = wvalid.to(device)
            rfeats   = rfeats.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            logits = model(wprobs, woffs, wvalid, rfeats).squeeze(-1)  # (N,)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tlosses.append(loss.item())

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                sys.stderr.write(
                    f"ReadCalib Epoch [{epoch+1}/{args.max_epoch_num}], "
                    f"Step [{i+1}/{total_step}]; "
                    f"TrainLoss: {np.mean(tlosses):.6f}; "
                    f"Time: {time.time()-start:.2f}s\n"
                )
                start = time.time()
                tlosses = []

        model.eval()
        with torch.no_grad():
            vlosses, val_preds, val_labels = [], [], []
            for wprobs, woffs, wvalid, rfeats, labels in valid_loader:
                wprobs  = wprobs.to(device)
                woffs   = woffs.to(device)
                wvalid  = wvalid.to(device)
                rfeats  = rfeats.to(device)
                labels  = labels.to(device)
                logits  = model(wprobs, woffs, wvalid, rfeats).squeeze(-1)
                vlosses.append(criterion(logits, labels).item())
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        v_meanloss = np.mean(vlosses)
        val_preds  = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc    = np.mean((val_preds > 0.5) == val_labels)
        pcc, _     = pearsonr(val_preds, val_labels)

        clean_state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        save_path = model_dir + f"read_calib.{args.read_calib_model_type}.b{args.read_calib_seq_len}_epoch{epoch+1}.ckpt"
        torch.save(clean_state, save_path)

        if v_meanloss < curr_lowest_loss:
            curr_lowest_loss = v_meanloss
            no_improve_count = 0
            curr_best_epoch = epoch + 1
            torch.save(clean_state, model_dir + "read_calib.best.ckpt")
        else:
            no_improve_count += 1

        cur_lr = optimizer.param_groups[0]['lr']
        sys.stderr.write(
            f"ReadCalib Epoch [{epoch+1}/{args.max_epoch_num}]; LR: {cur_lr:.2e}; "
            f"ValidLoss: {v_meanloss:.6f} | Acc: {val_acc:.4f} | PCC: {pcc:.4f}\n"
            f"BestValLoss: {curr_lowest_loss:.6f} (epoch {curr_best_epoch}) | "
            f"NoImprove: {no_improve_count}/{patience}\n"
        )

        model.train()

        if no_improve_count >= patience and epoch >= args.min_epoch_num - 1:
            sys.stderr.write(f"[read_calib-cpu] Early stop at epoch {epoch+1}\n")
            break

        if args.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(v_meanloss)
        else:
            scheduler.step()

    sys.stderr.write(
        f"✅ Best ReadCalib model at epoch {curr_best_epoch} "
        f"(ValLoss: {curr_lowest_loss:.6f})\n"
    )
    clear_linecache()


def train_multigpu(args):
    total_start = time.time()
    torch.manual_seed(args.tseed)

    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    if args.model_class == "aggregate" and not use_cuda:
        train_aggregate_cpu(args)
        return

    if args.model_class == "read_calib" and not use_cuda:
        train_read_calib_cpu(args)
        return

    if use_cuda:
        print("GPU is available!")
    else:
        raise RuntimeError("No GPU is available!")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available!")

    if torch.cuda.device_count() < args.ngpus_per_node:
        raise RuntimeError("There are not enough gpus, has {}, request {}.".format(torch.cuda.device_count(),
                                                                                   args.ngpus_per_node))

    global_world_size = args.ngpus_per_node * args.nodes

    if args.model_class == "mtm":
        worker_fn = train_worker_mtm
    elif args.model_class == "aggregate":
        worker_fn = train_worker_aggregate
    else:
        worker_fn = train_worker
    mp.spawn(worker_fn, nprocs=args.ngpus_per_node, args=(global_world_size, args))

    endtime = time.time()
    clear_linecache()
    print("[main]train_multigpu costs {:.1f} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser("[EXPERIMENTAL]train a model, use torch.nn.parallel.DistributedDataParallel")
    st_input = parser.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = parser.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=True)

    st_train = parser.add_argument_group("TRAIN MODEL_HYPER")
    st_train.add_argument(
        "--model_class",
        type=str,
        default="bilstm",
        choices=["bilstm", "mtm", "aggregate"],
        required=False,
        help="model class: 'bilstm' (ModelBiLSTM), 'mtm' (modelMTM), "
             "'aggregate' (site-level AggrAttRNN). default: bilstm",
    )
    st_train.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    st_train.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    # model param
    st_train.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    st_train.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    st_train.add_argument("--class_num", type=int, default=2, required=False)
    st_train.add_argument("--dropout_rate", type=float, default=0.5, required=False)
    st_train.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    st_train.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    st_train.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    st_train.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    st_train.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )
    # BiLSTM model param
    st_train.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )

    st_training = parser.add_argument_group("TRAINING")
    # model training
    st_training.add_argument('--optim_type', type=str, default="Adam",
                             choices=["Adam", "AdamW", "RMSprop", "SGD", "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, default Adam")
    st_training.add_argument('--batch_size', type=int, default=512, required=False)
    st_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                             choices=["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
                             help="StepLR, ReduceLROnPlateau or CosineAnnealingLR, default StepLR")
    st_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001. [lr should be lr*world_size when using multi gpus? "
                                  "or lower batch_size?]")
    st_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    st_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    st_training.add_argument('--lr_patience', type=int, default=0, required=False,
                             help="effective in ReduceLROnPlateau. default 0")
    st_training.add_argument("--max_epoch_num", action="store", default=20, type=int,
                             required=False, help="max epoch num, default 20")
    st_training.add_argument("--min_epoch_num", action="store", default=5, type=int,
                             required=False, help="min epoch num, default 5")
    st_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_training.add_argument('--step_interval', type=int, default=500, required=False)
    st_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")

    st_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    st_training.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')
    st_training.add_argument('--use_compile', type=str, default="no", required=False,
                             help="[EXPERIMENTAL] if using torch.compile, yes or no, "
                                  "default no ('yes' only works in pytorch>=2.0)")
    st_training.add_argument('--patience', type=int, default=3, required=False,
                             help="early stopping patience (epochs without improvement), default 5")

    st_mtm = parser.add_argument_group("MTM MODEL_HYPER (--model_class mtm)")
    st_mtm.add_argument('--mtm_num_base_features', type=int, default=1, required=False,
                        help="number of raw signal features per base position, default 1")
    st_mtm.add_argument('--mtm_hid_rnn', type=int, default=128, required=False,
                        help="d_model (hidden size) for MTM, default 128")
    st_mtm.add_argument('--mtm_d_static', type=int, default=1, required=False,
                        help="dimension of static feature vector (d_static), default 1")
    st_mtm.add_argument('--mtm_ratios', type=int, nargs='+', default=[2, 2, 2, 2], required=False,
                        help="downsampling ratios for MTM, e.g. --mtm_ratios 2 2 2 2, default [2, 2, 2, 2]")
    st_mtm.add_argument('--mtm_r_hid', type=int, default=4, required=False,
                        help="hidden ratio in TokenMixingLayer MLP, default 4")
    st_mtm.add_argument('--mtm_norm_first', type=str, default="True", required=False,
                        help="pre-norm (True) or post-norm (False) in TokenMixingLayer, default True")
    st_mtm.add_argument('--mtm_down_mode', type=str, default="concat",
                        choices=["concat", "avg", "max"], required=False,
                        help="downsampling aggregation mode, default concat")
    st_mtm.add_argument('--mtm_temporal_depth', type=int, default=2, required=False,
                        help="number of temporal attention layers per TokenMixingLayer, default 2")
    st_mtm.add_argument('--offset', type=int, default=0, required=False,
                        help="offset parameter recorded in checkpoint filename, default 0")

    st_agg = parser.add_argument_group("AGGREGATE MODEL_HYPER (--model_class aggregate)")
    st_agg.add_argument('--aggregate_model_type', type=str, default="attbigru",
                        choices=["attbigru", "transformer"], required=False,
                        help="aggregate model architecture, default attbigru")
    st_agg.add_argument('--aggregate_hidden', type=int, default=32, required=False,
                        help="hidden size for aggregate model, default 32")

    st_rc = parser.add_argument_group("READ_CALIB MODEL_HYPER (--model_class read_calib)")
    st_rc.add_argument('--read_calib_model_type', type=str, default="attbigru",
                       choices=["attbigru", "attbilstm"], required=False,
                       help="ReadCalibRNN architecture, default attbigru")
    st_rc.add_argument('--read_calib_hidden', type=int, default=32, required=False,
                       help="hidden size for ReadCalibRNN, default 32")
    st_rc.add_argument('--read_calib_seq_len', type=int, default=11, required=False,
                       help="K window size (CpG sites per read context), default 11")

    st_trainingp = parser.add_argument_group("TRAINING PARALLEL")
    st_trainingp.add_argument("--nodes", default=1, type=int,
                              help="number of nodes for distributed training, default 1")
    st_trainingp.add_argument("--ngpus_per_node", default=2, type=int,
                              help="number of GPUs per node for distributed training, default 2")
    st_trainingp.add_argument("--dist-url", default="tcp://127.0.0.1:12315", type=str,
                              help="url used to set up distributed training")
    st_trainingp.add_argument("--node_rank", default=0, type=int,
                              help="node rank for distributed training, default 0")
    st_trainingp.add_argument("--epoch_sync", action="store_true", default=False,
                              help="if sync model params of gpu0 to other local gpus after per epoch")
    
    args = parser.parse_args()

    display_args(args)
    train_multigpu(args)


if __name__ == "__main__":
    main()
