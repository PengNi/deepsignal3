"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_loc,
prob_0, prob_1, called_label, seq
"""

from __future__ import absolute_import

import os
# add this export temporarily
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import argparse
import sys
import numpy as np
from sklearn import metrics

import gzip

# import multiprocessing as mp
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# TODO: when using below import, will raise AttributeError: 'Queue' object has no attribute '_size'
# TODO: didn't figure out why
# from .utils.process_utils import Queue
from torch.multiprocessing import Queue

from torch.utils.data import DataLoader,DistributedSampler
from torch.cuda.amp import autocast

import torch.distributed as dist

import time

from .models import ModelBiLSTM
from .utils.process_utils import base2code_dna
from .utils.process_utils import code2base_dna
from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import normalize_signals
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import get_files
from .utils.process_utils import fill_files_queue
from .utils.process_utils import read_position_file

from .extract_features import _extract_preprocess

from .utils.constants_torch import FloatTensor
from .utils.constants_torch import use_cuda

import uuid

from .extract_features import _get_read_sequened_strand
from .extract_features import get_aligner
from .extract_features import start_extract_processes
from .extract_features import start_map_threads
from .extract_features import _reads_processed_stats
from .extract_features import _group_signals_by_movetable_v2
from .extract_features import _get_signals_rect
#from .extract_features_pod5 import match_pod5_and_bam

from .utils_dataloader import Pod5Dataset,Fast5Dataset,TsvDataset

from .utils_dataloader import collate_fn_inference
from .utils_dataloader import worker_init_fn

from .utils.process_utils import get_logger
from .utils.process_utils import _get_gpus

from .utils import bam_reader

import pod5
import mappy
import threading
import warnings

LOGGER = get_logger(__name__)

# add this export temporarily
os.environ["MKL_THREADING_LAYER"] = "GNU"

queue_size_border = 2000
qsize_size_border_p5batch = 40
queue_size_border_f5batch = 100
time_wait = 0.01
key_sep = "||"






def call_mods(args):
    start = time.time()
    LOGGER.info("[call_mods] starts")
    
    # Validate paths
    model_path = validate_path(args.model_path, "--model_path")
    input_path = validate_path(args.input_path, "--input_path")
    
    success_file = prepare_success_file(input_path)

    # Handle directory input (for pod5 or fast5)
    if os.path.isdir(input_path):
        handle_directory_input(args, input_path, model_path, success_file)
    else:
        handle_file_input(args, input_path, model_path, success_file)

    # Clean up
    cleanup_success_file(success_file)
    LOGGER.info("[call_mods] costs %.2f seconds.." % (time.time() - start))

def validate_path(path, error_message):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise ValueError(f"{error_message} is not set right!")
    return abs_path

def prepare_success_file(input_path):
    success_file = input_path.rstrip("/") + "." + str(uuid.uuid1()) + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)
    return success_file

def handle_directory_input(args, input_path, model_path, success_file):
    ref_path = validate_reference_path(args.reference_path) if args.reference_path else None
    is_dna = not args.rna
    is_recursive = str2bool(args.recursively)

    if args.pod5:
        handle_pod5_input(args, input_path, model_path, success_file, is_dna, is_recursive)
    else:
        handle_fast5_input(args, input_path, model_path, success_file, ref_path, is_dna, is_recursive)

def validate_reference_path(ref_path):
    return validate_path(ref_path, "--reference_path")

def handle_pod5_input(args, input_path, model_path, success_file, is_dna, is_recursive):
    bam_index = bam_reader.ReadIndexedBam(args.bam)
    motif_seqs = get_motif_seqs(args.motifs, is_dna)
    positions = read_position_file(args.positions) if args.positions else None
    pod5_dr = get_files(input_path, is_recursive, ".pod5")

    if use_cuda:
        _call_mods_from_pod5_gpu(pod5_dr, bam_index, success_file, model_path, motif_seqs, positions, args)
    else:
        _call_mods_from_pod5_cpu(pod5_dr, bam_index, success_file, model_path, motif_seqs, positions, args)

def handle_fast5_input(args, input_path, model_path, success_file, ref_path, is_dna, is_recursive):
    read_strand = _get_read_sequened_strand(args.basecall_subgroup)
    motif_seqs, chrom2len, _, len_fast5s, positions, contigs = _extract_preprocess(
        input_path, is_recursive, args.motifs, is_dna, ref_path, args.r_batch_size, args.positions, args
    )
    aligner=None
    fast5s_q = get_files(input_path, is_recursive, ".fast5")
    if args.mapping:
        aligner = get_aligner(ref_path, args.best_n)

    if use_cuda:
        _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args,aligner)
    else:
        _call_mods_from_fast5s_cpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args,aligner)

def handle_file_input(args, input_path, model_path, success_file):
    if use_cuda:
        _call_mods_from_file_gpu(input_path, model_path, args)
    

def determine_process_count(args):
    if use_cuda:
        return max(1, args.nproc)
    else:
        nproc = max(1, args.nproc)
        #nproc_dp = min(nproc - 2, nproc_to_call_mods_in_cpu_mode)
        return nproc

def cleanup_success_file(success_file):
    if os.path.exists(success_file):
        os.remove(success_file)

def load_model(model_path: str,device, args):
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
    )
    try:
        para_dict = torch.load(model_path, map_location=torch.device(device))
    except Exception:
        para_dict = torch.jit.load(model_path)
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)
    del model_dict
    # with warnings.catch_warnings():
    #     model = ModelBiLSTM.load_from_checkpoint(model_path,
    #                                           strict=False,
    #                                           track_metrics=False)
    if use_cuda:
        gpulist = _get_gpus()  # 获取可用GPU列表
        gpuindex = 0
        model = torch.nn.DataParallel(model,device_ids=gpulist,)#output_device=gpulist[len(gpulist)-1]
        model = model.to(device)

    return model

def _call_mods_from_pod5_gpu(
    pod5_dr, bam_index, success_file, model_path, motif_seqs, positions, args
):
    
    # dist.init_process_group(backend='nccl',init_method="tcp://127.0.0.1:12315",rank=0,world_size=4)
    # local_rank, world_size = dist.get_rank(), dist.get_world_size()
    # LOGGER.info(f"local_rank: {local_rank}, world_size: {world_size}")
    
    nproc = determine_process_count(args)
    gpus=_get_gpus()
    gpuindex = 0
    device = torch.device(f"cuda:{gpus[gpuindex]}" if torch.cuda.is_available() else "cpu")
    dataset = Pod5Dataset(pod5_dr, bam_index, motif_seqs, positions, device, args)
    #sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                        worker_init_fn=worker_init_fn,collate_fn=collate_fn_inference,#sampler=sampler,
                        pin_memory=True
                        )
    
    # 创建两个队列：一个用于传递特征批次，一个用于存储预测字符串
    # features_batch_q = Queue()
    pred_str_q = Queue()

    # 启动写进程，将预测结果写入文件
    p_w = mp.Process(
        target=_write_predstr_to_file,
        args=(args.result_file, pred_str_q),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()
    model=load_model(model_path,device, args)

    # 初始化模型并设置到GPU
    
    model.eval()
    with torch.no_grad() and torch.cuda.amp.autocast():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            
            pred_str, accuracy, batch_num = _call_mods(batch,model,args.batch_size)
            if pred_str ==[]:
                print("pred_str is empty")
                continue
            #pred_str=['test']
            pred_str_q.put(pred_str)

    # 在处理完成后，发送结束信号
    # features_batch_q.put("kill")
    pred_str_q.put("kill")

    # # 等待所有进程结束
    # for p in predstr_procs:
    #     p.join()

    p_w.join()

def _call_mods_from_pod5_cpu(
    pod5_dr, bam_index, success_file, model_path, motif_seqs, positions, args
):
    nproc = determine_process_count(args)
    dataset = Pod5Dataset(pod5_dr, bam_index, motif_seqs, positions, 'cpu', args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                        worker_init_fn=worker_init_fn,collate_fn=collate_fn_inference,
                        #pin_memory=True
                        )
    
    # 创建两个队列：一个用于传递特征批次，一个用于存储预测字符串
    # features_batch_q = Queue()
    pred_str_q = Queue()

    # 启动写进程，将预测结果写入文件
    p_w = mp.Process(
        target=_write_predstr_to_file,
        args=(args.result_file, pred_str_q),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()
    model=load_model(model_path,'cpu', args)

    # 初始化模型并设置到GPU
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            pred_str, accuracy, batch_num = _call_mods(batch,model,args.batch_size)
            if pred_str ==[]:
                print("pred_str is empty")
                continue
            #pred_str=['test']
            pred_str_q.put(pred_str)

    # 在处理完成后，发送结束信号
    # features_batch_q.put("kill")
    pred_str_q.put("kill")

    # # 等待所有进程结束
    # for p in predstr_procs:
    #     p.join()

    p_w.join()

def _call_mods_from_fast5s_gpu(
    ref_path,
    motif_seqs,
    chrom2len,
    fast5s_q,
    len_fast5s,
    positions,
    chrom2seqs,
    model_path,
    success_file,
    read_strand,
    args,
    aligner
):
    nproc = determine_process_count(args)
    gpus=_get_gpus()
    gpuindex = 0
    device = torch.device(f"cuda:{gpus[gpuindex]}" if torch.cuda.is_available() else "cpu")
    dataset = Fast5Dataset(fast5s_q, motif_seqs, positions, device, args, chrom2len, read_strand,chrom2seqs, aligner)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                        worker_init_fn=worker_init_fn,collate_fn=collate_fn_inference,
                        #pin_memory=True
                        )
    
    # 创建两个队列：一个用于传递特征批次，一个用于存储预测字符串
    # features_batch_q = Queue()
    pred_str_q = Queue()

    # 启动写进程，将预测结果写入文件
    p_w = mp.Process(
        target=_write_predstr_to_file,
        args=(args.result_file, pred_str_q),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()
    model=load_model(model_path,device, args)

    # 初始化模型并设置到GPU
    
    model.eval()
    with torch.no_grad() and torch.cuda.amp.autocast():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            pred_str, accuracy, batch_num = _call_mods(batch,model,args.batch_size)
            if pred_str ==[]:
                print("pred_str is empty")
                continue
            #pred_str=['test']
            pred_str_q.put(pred_str)

    # 在处理完成后，发送结束信号
    # features_batch_q.put("kill")
    pred_str_q.put("kill")

    # # 等待所有进程结束
    # for p in predstr_procs:
    #     p.join()

    p_w.join()


def _call_mods_from_fast5s_cpu(
    ref_path,
    motif_seqs,
    chrom2len,
    fast5s_q,
    len_fast5s,
    positions,
    chrom2seqs,
    model_path,
    success_file,
    read_strand,
    args,
):
    pass

def _call_mods_from_file_gpu(
     input_path, model_path, args
):
    
    # dist.init_process_group(backend='nccl',init_method="tcp://127.0.0.1:12315",rank=0,world_size=4)
    # local_rank, world_size = dist.get_rank(), dist.get_world_size()
    # LOGGER.info(f"local_rank: {local_rank}, world_size: {world_size}")
    
    nproc = determine_process_count(args)
    gpus=_get_gpus()
    gpuindex = 0
    device = torch.device(f"cuda:{gpus[gpuindex]}" if torch.cuda.is_available() else "cpu")
    dataset = TsvDataset(input_path)
    #sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                        worker_init_fn=worker_init_fn,collate_fn=collate_fn_inference,#sampler=sampler,
                        pin_memory=True
                        )
    
    # 创建两个队列：一个用于传递特征批次，一个用于存储预测字符串
    # features_batch_q = Queue()
    pred_str_q = Queue()

    # 启动写进程，将预测结果写入文件
    p_w = mp.Process(
        target=_write_predstr_to_file,
        args=(args.result_file, pred_str_q),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()
    model=load_model(model_path,device, args)

    # 初始化模型并设置到GPU
    
    model.eval()
    with torch.no_grad() and torch.cuda.amp.autocast():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            
            pred_str, accuracy, batch_num = _call_mods(batch,model,args.batch_size)
            if pred_str ==[]:
                print("pred_str is empty")
                continue
            #pred_str=['test']
            pred_str_q.put(pred_str)

    # 在处理完成后，发送结束信号
    # features_batch_q.put("kill")
    pred_str_q.put("kill")

    # # 等待所有进程结束
    # for p in predstr_procs:
    #     p.join()

    p_w.join()

def _call_mods(features_batch, model, batch_size, device=0):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples (not any more, modified)
    # --------------: 2. if from _read_features_from_fast5s(), has uncertain number of samples
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, k_signals, labels = features_batch#zip(*features_batch)
    #print(kmers[0])
    #return (kmers,1,1)
    #labels = np.reshape(labels, (len(labels)))
    # print('labels shape: {}'.format(labels.shape))
    # nkmers=np.array(kmers)
    # print('nkmers shape: {}'.format(nkmers.shape))
    # print('nkmers: {}'.format(nkmers[0]))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]
        b_kmers = kmers[batch_s:batch_e]
        b_base_means = base_means[batch_s:batch_e]
        b_base_stds = base_stds[batch_s:batch_e]
        b_base_signal_lens = base_signal_lens[batch_s:batch_e]
        # b_base_probs = base_probs[batch_s:batch_e]
        b_k_signals = k_signals[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            _, vlogits = model(
                b_kmers.cuda(non_blocking=True),
                b_base_means.cuda(non_blocking=True),
                b_base_stds.cuda(non_blocking=True),
                b_base_signal_lens.cuda(non_blocking=True),
                b_k_signals.cuda(non_blocking=True),
            )
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            acc_batch = metrics.accuracy_score(y_true=b_labels, y_pred=predicted)
            accuracys.append(acc_batch)

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, pos_in_strand, read_name, read_strand, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(1 - prob_0_norm, 6)
                # kmer-5
                b_idx_kmer = "".join([code2base_dna[int(x)] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer) / 2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = (
                    center_idx + 3
                    if center_idx + 3 <= len(b_idx_kmer)
                    else len(b_idx_kmer)
                )

                pred_str.append(
                    "\t".join(
                        [
                            b_sampleinfo[idx],
                            str(prob_0_norm),
                            str(prob_1_norm),
                            str(predicted[idx]),
                            b_idx_kmer[bkmer_start:bkmer_end],
                        ]
                    )
                )
            batch_num += 1
    accuracy = np.mean(accuracys) if len(accuracys) > 0 else 0

    return pred_str, accuracy, batch_num

def _write_predstr_to_file(write_fp, predstr_q):
    LOGGER.info("write_process-{} starts".format(os.getpid()))
    with open(write_fp, "w") as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                LOGGER.info("write_process-{} finished".format(os.getpid()))
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument(
        "--input_path",
        "-i",
        action="store",
        type=str,
        required=True,
        help="the input path, can be a signal_feature file from extract_features.py, "
        "or a directory of fast5 files. If a directory of fast5 files is provided, "
        "args in FAST5_EXTRACTION and MAPPING should (reference_path must) be provided.",
    )
    p_input.add_argument(
        "--r_batch_size",
        action="store",
        type=int,
        default=50,
        required=False,
        help="number of files to be processed by each process one time, default 50",
    )
    p_input.add_argument(
        "--pod5",
        action="store_true",
        default=False,
        required=False,
        help="use pod5, default false",
    )
    p_input.add_argument("--bam", type=str, help="the bam filepath")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument(
        "--model_path",
        "-m",
        action="store",
        type=str,
        required=True,
        help="file path of the trained model (.ckpt)",
    )

    # model input
    p_call.add_argument(
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
    )
    p_call.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    p_call.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="signal num of one base, default 15",
    )

    # model param
    p_call.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    p_call.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    p_call.add_argument("--class_num", type=int, default=2, required=False)
    p_call.add_argument("--dropout_rate", type=float, default=0, required=False)
    p_call.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    p_call.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    p_call.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    p_call.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    p_call.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )

    p_call.add_argument(
        "--batch_size",
        "-b",
        default=512,
        type=int,
        required=False,
        action="store",
        help="batch size, default 512",
    )

    # BiLSTM model param
    p_call.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument(
        "--result_file",
        "-o",
        action="store",
        type=str,
        required=True,
        help="the file path to save the predicted result",
    )

    p_f5 = parser.add_argument_group("EXTRACTION")
    p_f5.add_argument(
        "--single",
        action="store_true",
        default=False,
        required=False,
        help="the fast5 files are in single-read format",
    )
    p_f5.add_argument(
        "--recursively",
        "-r",
        action="store",
        type=str,
        required=False,
        default="yes",
        help="is to find fast5 files from fast5 dir recursively. "
        "default true, t, yes, 1",
    )
    p_f5.add_argument(
        "--rna",
        action="store_true",
        default=False,
        required=False,
        help="the fast5 files are from RNA samples. if is rna, the signals are reversed. "
        "NOTE: Currently no use, waiting for further extentsion",
    )
    p_f5.add_argument(
        "--basecall_group",
        action="store",
        type=str,
        required=False,
        default=None,
        help="basecall group generated by Guppy. e.g., Basecall_1D_000",
    )
    p_f5.add_argument(
        "--basecall_subgroup",
        action="store",
        type=str,
        required=False,
        default="BaseCalled_template",
        help="the basecall subgroup of fast5 files. default BaseCalled_template",
    )
    p_f5.add_argument(
        "--reference_path",
        action="store",
        type=str,
        required=False,
        help="the reference file to be used, usually is a .fa file",
    )
    p_f5.add_argument(
        "--normalize_method",
        action="store",
        type=str,
        choices=["mad", "zscore"],
        default="mad",
        required=False,
        help="the way for normalizing signals in read level. "
        "mad or zscore, default mad",
    )
    p_f5.add_argument(
        "--methy_label",
        action="store",
        type=int,
        choices=[1, 0],
        required=False,
        default=1,
        help="the label of the interested modified bases, this is for training."
        " 0 or 1, default 1",
    )
    p_f5.add_argument(
        "--motifs",
        action="store",
        type=str,
        required=False,
        default="CG",
        help="motif seq to be extracted, default: CG. "
        "can be multi motifs splited by comma "
        "(no space allowed in the input str), "
        "or use IUPAC alphabet, "
        "the mod_loc of all motifs must be "
        "the same",
    )
    p_f5.add_argument(
        "--mod_loc",
        action="store",
        type=int,
        required=False,
        default=0,
        help="0-based location of the targeted base in the motif, default 0",
    )
    p_f5.add_argument(
        "--pad_only_r",
        action="store_true",
        default=False,
        help="pad zeros to only the right of signals array of one base, "
        "when the number of signals is less than --signal_len. "
        "default False (pad in two sides).",
    )
    p_f5.add_argument(
        "--positions",
        action="store",
        type=str,
        required=False,
        default=None,
        help="file with a list of positions interested (must be formatted as tab-separated file"
        " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
        "need to be set. --positions is used to narrow down the range of the trageted "
        "motif locs. default None",
    )
    p_f5.add_argument(
        "--trace",
        action="store_true",
        default=False,
        required=False,
        help="use trace, default false",
    )
    p_mape = parser.add_argument_group("MAPe")
    p_mape.add_argument(
        "--corrected_group",
        action="store",
        type=str,
        required=False,
        default="RawGenomeCorrected_000",
        help="the corrected_group of fast5 files, " "default RawGenomeCorrected_000",
    )

    p_mapping = parser.add_argument_group("MAPPING")
    p_mapping.add_argument(
        "--mapping",
        action="store_true",
        default=False,
        required=False,
        help="use MAPPING to get alignment, default false",
    )
    p_mapping.add_argument(
        "--mapq",
        type=int,
        default=1,
        required=False,
        help="MAPping Quality cutoff for selecting alignment items, default 1",
    )
    p_mapping.add_argument(
        "--identity",
        type=float,
        default=0.0,
        required=False,
        help="identity cutoff for selecting alignment items, default 0.0",
    )
    p_mapping.add_argument(
        "--coverage_ratio",
        type=float,
        default=0.50,
        required=False,
        help="percent of coverage, read alignment len against read len, default 0.50",
    )
    p_mapping.add_argument(
        "--best_n",
        "-n",
        type=int,
        default=1,
        required=False,
        help="best_n arg in mappy(minimap2), default 1",
    )

    parser.add_argument(
        "--nproc",
        "-p",
        action="store",
        type=int,
        default=10,
        required=False,
        help="number of processes to be used, default 10.",
    )
    parser.add_argument(
        "--nproc_gpu",
        action="store",
        type=int,
        default=2,
        required=False,
        help="number of processes to use gpu (if gpu is available), "
        "1 or a number less than nproc-1, no more than "
        "nproc/4 is suggested. default 2.",
    )

    args = parser.parse_args()
    display_args(args)
    call_mods(args)


if __name__ == "__main__":
    sys.exit(main())
