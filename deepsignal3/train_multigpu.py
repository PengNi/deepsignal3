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
)
from .dataloader import (
    SignalFeaData1s,
    generate_offsets,
)
from .dataloader import clear_linecache
from .utils.process_utils import display_args
from .utils.process_utils import str2bool
from .utils.process_utils import count_line_num

from .utils.constants_torch import use_cuda

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# add this export temporarily
# https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'


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
                        r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
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
        args.model_type,
        local_rank
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
                                      patience=args.lr_patience, verbose=True)
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
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels = (
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
            for vi, vsfeatures in enumerate(valid_loader):
                (
                    _,
                    vkmer,
                    vbase_means,
                    vbase_stds,
                    vbase_signal_lens,
                    vsignals,
                    vlabels,
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

                dist.barrier()
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
                               model_dir + args.model_type +
                               '.b{}_s{}_epoch{}.ckpt'.format(args.seq_len, args.signal_len, epoch + 1))
                # TODO: dist.barrier()? and read/sync model dict?
                if v_accuracy > curr_best_accuracy:
                    curr_best_accuracy = v_accuracy
                    curr_best_accuracy_loc = epoch + 1

                if len(v_accuracy_epoches) > 0 and v_accuracy > \
                        v_accuracy_epoches[-1]:
                    if global_rank == 0:
                        torch.save(model.module.state_dict(),
                                   model_dir + args.model_type +
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
                    last_lr = scheduler.get_last_lr()
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
            sync_ckpt = model_dir + args.model_type + \
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


def train_multigpu(args):
    total_start = time.time()
    torch.manual_seed(args.tseed)
    
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

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
    mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(global_world_size, args))

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
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
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
    st_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                  "'Ranger' or 'LookaheadAdam', default Adam")
    st_training.add_argument('--batch_size', type=int, default=512, required=False)
    st_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                             choices=["StepLR", "ReduceLROnPlateau"],
                             help="StepLR or ReduceLROnPlateau, default StepLR")
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
