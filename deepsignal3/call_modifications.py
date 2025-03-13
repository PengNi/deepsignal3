"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_loc,
prob_0, prob_1, called_label, seq
"""

from __future__ import absolute_import

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import argparse
import sys
import numpy as np
from sklearn import metrics
import gzip
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from torch.multiprocessing import Queue
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
import pod5
import pyslow5

from .models import ModelBiLSTM
from .utils.process_utils import base2code_dna, code2base_dna, str2bool, display_args, normalize_signals
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode, get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs, get_files, fill_files_queue, read_position_file,detect_file_type,validate_path
from .extract_features import _extract_preprocess
from .utils.constants_torch import FloatTensor, use_cuda
from .extract_features import _get_read_sequened_strand, get_aligner
from .extract_features import _group_signals_by_movetable_v2, _get_signals_rect
from .utils_dataloader import SignalDataset, Fast5Dataset, TsvDataset
from .utils_dataloader import collate_fn_inference, worker_init_fn
from .utils.process_utils import get_logger, _get_gpus
from .utils import bam_reader
import mappy
import threading
import warnings
import uuid

warnings.filterwarnings("ignore", category=FutureWarning)

LOGGER = get_logger(__name__)
os.environ["MKL_THREADING_LAYER"] = "GNU"

queue_size_border = 2000
qsize_size_border_p5batch = 40
queue_size_border_f5batch = 100
time_wait = 0.01
key_sep = "||"

def call_mods(args):
    start = time.time()
    LOGGER.info("[call_mods] starts")
    
    model_path = validate_path(args.model_path, "--model_path")
    input_path = validate_path(args.input_path, "--input_path")
    success_file = prepare_success_file(input_path)
    
    file_type = detect_file_type(input_path, str2bool(args.recursively))
    if file_type in ['pod5', 'slow5', 'fast5']:
        handle_signal_or_fast5_input(args, input_path, model_path, success_file, file_type)
    else:
        handle_file_input(args, input_path, model_path, success_file)  # 处理 TSV 文件

    cleanup_success_file(success_file)
    LOGGER.info("[call_mods] costs %.2f seconds.." % (time.time() - start))


def prepare_success_file(input_path):
    success_file = input_path.rstrip("/") + "." + str(uuid.uuid1()) + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)
    return success_file

def handle_signal_or_fast5_input(args, input_path, model_path, success_file, file_type):
    ref_path = validate_reference_path(args.reference_path) if args.reference_path else None
    is_dna = not args.rna
    is_recursive = str2bool(args.recursively)
    
    if file_type in ['pod5', 'slow5']:
        bam_index = bam_reader.ReadIndexedBam(args.bam)
        motif_seqs = get_motif_seqs(args.motifs, is_dna)
        positions = read_position_file(args.positions) if args.positions else None
        files_dr = get_files(input_path, is_recursive, ".pod5" if file_type == 'pod5' else (".slow5", ".blow5"))
        
        files_queue = Queue()
        fill_files_queue(files_queue, files_dr)
        
        if use_cuda:
            pred_str_q = Queue()
            p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q), name="writer")
            p_w.daemon = True
            p_w.start()
            world_size = torch.cuda.device_count()
            param = (files_dr, bam_index, success_file, model_path, motif_seqs, positions, pred_str_q, files_queue, args, file_type)
            mp.spawn(_call_mods_from_signal_gpu_distributed, args=(world_size, param), nprocs=world_size, join=True)
            pred_str_q.put("kill")
            p_w.join()
        else:
            _call_mods_from_signal_cpu(files_dr, bam_index, success_file, model_path, motif_seqs, positions, files_queue, args, file_type)
    elif file_type == 'fast5':
        read_strand = _get_read_sequened_strand(args.basecall_subgroup)
        motif_seqs, chrom2len, _, len_fast5s, positions, contigs = _extract_preprocess(
            input_path, is_recursive, args.motifs, is_dna, ref_path, args.r_batch_size, args.positions, args
        )
        aligner = get_aligner(ref_path, args.best_n) if args.mapping else None
        fast5s_q = get_files(input_path, is_recursive, ".fast5")
        
        if use_cuda:
            _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args, aligner)
        else:
            _call_mods_from_fast5s_cpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args, aligner)

def handle_directory_input(args, input_path, model_path, success_file):
    ref_path = validate_reference_path(args.reference_path) if args.reference_path else None
    is_dna = not args.rna
    is_recursive = str2bool(args.recursively)
    file_type = detect_file_type(input_path, is_recursive)
    
    if file_type in ['pod5', 'slow5']:
        handle_signal_input(args, input_path, model_path, success_file, is_dna, is_recursive, file_type)
    elif file_type == 'fast5':
        handle_fast5_input(args, input_path, model_path, success_file, ref_path, is_dna, is_recursive)
    else:
        raise ValueError(f"No valid signal files (.pod5, .slow5, .blow5, .fast5) found in {input_path}")

def validate_reference_path(ref_path):
    return validate_path(ref_path, "--reference_path")

def handle_signal_input(args, input_path, model_path, success_file, is_dna, is_recursive, file_type):
    bam_index = bam_reader.ReadIndexedBam(args.bam)
    motif_seqs = get_motif_seqs(args.motifs, is_dna)
    positions = read_position_file(args.positions) if args.positions else None
    files_dr = get_files(input_path, is_recursive, ".pod5" if file_type == 'pod5' else (".slow5", ".blow5"))
    
    files_queue = Queue()
    fill_files_queue(files_queue, files_dr)
    
    if use_cuda:
        pred_str_q = Queue()
        p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q), name="writer")
        p_w.daemon = True
        p_w.start()
        world_size = torch.cuda.device_count()
        param = (files_dr, bam_index, success_file, model_path, motif_seqs, positions, pred_str_q, files_queue, args, file_type)
        mp.spawn(_call_mods_from_signal_gpu_distributed, args=(world_size, param), nprocs=world_size, join=True)
        pred_str_q.put("kill")
        p_w.join()
    else:
        _call_mods_from_signal_cpu(files_dr, bam_index, success_file, model_path, motif_seqs, positions, files_queue, args, file_type)

def handle_fast5_input(args, input_path, model_path, success_file, ref_path, is_dna, is_recursive):
    read_strand = _get_read_sequened_strand(args.basecall_subgroup)
    motif_seqs, chrom2len, _, len_fast5s, positions, contigs = _extract_preprocess(
        input_path, is_recursive, args.motifs, is_dna, ref_path, args.r_batch_size, args.positions, args
    )
    aligner = get_aligner(ref_path, args.best_n) if args.mapping else None
    fast5s_q = get_files(input_path, is_recursive, ".fast5")
    
    if use_cuda:
        _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args, aligner)
    else:
        _call_mods_from_fast5s_cpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs, model_path, success_file, read_strand, args, aligner)

def handle_file_input(args, input_path, model_path, success_file):
    if use_cuda:
        _call_mods_from_file_gpu(input_path, model_path, args)

def determine_process_count(args):
    if use_cuda:
        return max(1, args.nproc)
    return max(1, args.nproc)

def cleanup_success_file(success_file):
    if os.path.exists(success_file):
        os.remove(success_file)

def load_model(model_path, device, args):
    model = ModelBiLSTM(
        args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
        args.dropout_rate, args.hid_rnn, args.n_vocab, args.n_embed, str2bool(args.is_base),
        str2bool(args.is_signallen), str2bool(args.is_trace), args.model_type
    )
    try:
        para_dict = torch.load(model_path, map_location=torch.device(device))
    except Exception:
        para_dict = torch.jit.load(model_path)
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)
    del model_dict
    if use_cuda:
        gpulist = _get_gpus()
        model = torch.nn.DataParallel(model, device_ids=gpulist)
        model = model.to(device)
    return model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_model_distributed(model_path, device, args):
    model = ModelBiLSTM(
        args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
        args.dropout_rate, args.hid_rnn, args.n_vocab, args.n_embed, str2bool(args.is_base),
        str2bool(args.is_signallen), str2bool(args.is_trace), args.model_type
    )
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")
    model.to(device)
    model = DDP(model, device_ids=[device.index])
    return model

def _call_mods_from_signal_gpu_distributed(rank, world_size, param):
    files_dr, bam_index, success_file, model_path, motif_seqs, positions, pred_str_q, files_queue, args, file_type = param
    setup(rank, world_size)
    LOGGER.info(f"Process {rank} initialized")
    nproc = determine_process_count(args)
    device = torch.device(f"cuda:{rank}")
    model = load_model_distributed(model_path, device, args)
    model.eval()
    
    dataset = SignalDataset(files_dr, bam_index, motif_seqs, positions, device, files_queue, args, format_type=file_type)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=nproc, collate_fn=collate_fn_inference,
        worker_init_fn=worker_init_fn, pin_memory=True
    )
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                LOGGER.info(f"Rank {rank}: batch is None")
                continue
            with autocast():
                pred_str, accuracy, batch_num = _call_mods(batch, model, args.batch_size)
            pred_str_q.put(pred_str)
    
    cleanup()

def _call_mods_from_signal_gpu(
    pod5_dr, bam_index, success_file, model_path, motif_seqs, positions, args, file_type
):
    
    # dist.init_process_group(backend='nccl',init_method="tcp://127.0.0.1:12315",rank=0,world_size=4)
    # local_rank, world_size = dist.get_rank(), dist.get_world_size()
    # LOGGER.info(f"local_rank: {local_rank}, world_size: {world_size}")
    
    nproc = determine_process_count(args)
    gpus=_get_gpus()
    gpuindex = 0
    device = torch.device(f"cuda:{gpus[gpuindex]}" if torch.cuda.is_available() else "cpu")
    dataset = SignalDataset(pod5_dr, bam_index, motif_seqs, positions, device, args, file_type)
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

def _call_mods_from_signal_cpu(files_dr, bam_index, success_file, model_path, motif_seqs, positions, files_queue, args, file_type):
    nproc = determine_process_count(args)
    dataset = SignalDataset(files_dr, bam_index, motif_seqs, positions, 'cpu', files_queue, args, format_type=file_type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                             worker_init_fn=worker_init_fn, collate_fn=collate_fn_inference)
    
    pred_str_q = Queue()
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q), name="writer")
    p_w.daemon = True
    p_w.start()
    model = load_model(model_path, 'cpu', args)
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            pred_str, accuracy, batch_num = _call_mods(batch, model, args.batch_size)
            if pred_str == []:
                print("pred_str is empty")
                continue
            pred_str_q.put(pred_str)
    
    pred_str_q.put("kill")
    p_w.join()

def _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, chrom2seqs, model_path, success_file, read_strand, args, aligner):
    nproc = determine_process_count(args)
    gpus = _get_gpus()
    device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
    dataset = Fast5Dataset(fast5s_q, motif_seqs, positions, device, args, chrom2len, read_strand, chrom2seqs, aligner)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                             worker_init_fn=worker_init_fn, collate_fn=collate_fn_inference, pin_memory=True)
    
    pred_str_q = Queue()
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q), name="writer")
    p_w.daemon = True
    p_w.start()
    model = load_model(model_path, device, args)
    model.eval()
    
    with torch.no_grad() and torch.cuda.amp.autocast():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            pred_str, accuracy, batch_num = _call_mods(batch, model, args.batch_size)
            if pred_str == []:
                print("pred_str is empty")
                continue
            pred_str_q.put(pred_str)
    
    pred_str_q.put("kill")
    p_w.join()

def _call_mods_from_fast5s_cpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, chrom2seqs, model_path, success_file, read_strand, args, aligner=None):
    pass  # Implement if needed

def _call_mods_from_file_gpu(input_path, model_path, args):
    nproc = determine_process_count(args)
    gpus = _get_gpus()
    device = torch.device(f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu")
    dataset = TsvDataset(input_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=nproc,
                             worker_init_fn=worker_init_fn, collate_fn=collate_fn_inference, pin_memory=True)
    
    pred_str_q = Queue()
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q), name="writer")
    p_w.daemon = True
    p_w.start()
    model = load_model(model_path, device, args)
    model.eval()
    
    with torch.no_grad() and torch.cuda.amp.autocast():
        for batch in data_loader:
            if batch is None:
                print("batch is None")
                continue
            pred_str, accuracy, batch_num = _call_mods(batch, model, args.batch_size)
            if pred_str == []:
                print("pred_str is empty")
                continue
            pred_str_q.put(pred_str)
    
    pred_str_q.put("kill")
    p_w.join()

def _call_mods(features_batch, model, batch_size, device=0):
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, k_signals, labels = features_batch
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
        b_k_signals = k_signals[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        
        if len(b_sampleinfo) > 0:
            _, _, _, vlogits = model(
                b_kmers.cuda(non_blocking=True), b_base_means.cuda(non_blocking=True),
                b_base_stds.cuda(non_blocking=True), b_base_signal_lens.cuda(non_blocking=True),
                b_k_signals.cuda(non_blocking=True)
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
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(1 - prob_0_norm, 6)
                b_idx_kmer = "".join([code2base_dna[int(x)] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer) / 2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)
                
                pred_str.append(
                    "\t".join([b_sampleinfo[idx], str(prob_0_norm), str(prob_1_norm), str(predicted[idx]), b_idx_kmer[bkmer_start:bkmer_end]])
                )
            batch_num += 1
    
    accuracy = np.mean(accuracys) if len(accuracys) > 0 else 0
    return pred_str, accuracy, batch_num

def _write_predstr_to_file(write_fp, predstr_q):
    LOGGER.info("write_process-{} starts".format(os.getpid()))
    with open(write_fp, "w") as wf:
        while True:
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
    p_input.add_argument("--input_path", "-i", type=str, required=True, help="the input path (signal_feature file or directory of fast5/pod5/slow5 files)")
    p_input.add_argument("--r_batch_size", type=int, default=50, help="number of files to process per batch, default 50")
    p_input.add_argument("--bam", type=str, help="the bam filepath")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_path", "-m", type=str, required=True, help="path to the trained model (.ckpt)")
    p_call.add_argument("--model_type", type=str, default="both_bilstm", choices=["both_bilstm", "seq_bilstm", "signal_bilstm"], help="type of model, default: both_bilstm")
    p_call.add_argument("--seq_len", type=int, default=21, help="len of kmer, default 21")
    p_call.add_argument("--signal_len", type=int, default=15, help="signal num of one base, default 15")
    p_call.add_argument("--layernum1", type=int, default=3, help="lstm layer num for combined feature, default 3")
    p_call.add_argument("--layernum2", type=int, default=1, help="lstm layer num for seq feature, default 1")
    p_call.add_argument("--class_num", type=int, default=2)
    p_call.add_argument("--dropout_rate", type=float, default=0)
    p_call.add_argument("--n_vocab", type=int, default=16, help="base_seq vocab_size, default 16")
    p_call.add_argument("--n_embed", type=int, default=4, help="base_seq embedding_size")
    p_call.add_argument("--is_base", type=str, default="yes", help="use base features in seq model, default yes")
    p_call.add_argument("--is_signallen", type=str, default="yes", help="use signal length feature, default yes")
    p_call.add_argument("--is_trace", type=str, default="no", help="use trace feature, default no")
    p_call.add_argument("--batch_size", "-b", type=int, default=512, help="batch size, default 512")
    p_call.add_argument("--hid_rnn", type=int, default=256, help="BiLSTM hidden_size, default 256")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--result_file", "-o", type=str, required=True, help="path to save the predicted result")

    p_f5 = parser.add_argument_group("EXTRACTION")
    p_f5.add_argument("--single", action="store_true", default=False, help="fast5 files are in single-read format")
    p_f5.add_argument("--recursively", "-r", type=str, default="yes", help="find files recursively, default yes")
    p_f5.add_argument("--rna", action="store_true", default=False, help="fast5 files are from RNA samples")
    p_f5.add_argument("--basecall_group", type=str, default=None, help="basecall group from Guppy")
    p_f5.add_argument("--basecall_subgroup", type=str, default="BaseCalled_template", help="basecall subgroup, default BaseCalled_template")
    p_f5.add_argument("--reference_path", type=str, help="reference file (.fa)")
    p_f5.add_argument("--normalize_method", type=str, choices=["mad", "zscore"], default="mad", help="signal normalization method, default mad")
    p_f5.add_argument("--methy_label", type=int, choices=[1, 0], default=1, help="label of modified bases, default 1")
    p_f5.add_argument("--motifs", type=str, default="CG", help="motif seq to extract, default CG")
    p_f5.add_argument("--mod_loc", type=int, default=0, help="0-based location of targeted base in motif, default 0")
    p_f5.add_argument("--pad_only_r", action="store_true", default=False, help="pad zeros only to right of signals")
    p_f5.add_argument("--positions", type=str, default=None, help="file with list of positions")
    p_f5.add_argument("--trace", action="store_true", default=False, help="use trace, default false")

    p_mape = parser.add_argument_group("MAPe")
    p_mape.add_argument("--corrected_group", type=str, default="RawGenomeCorrected_000", help="corrected_group of fast5 files")

    p_mapping = parser.add_argument_group("MAPPING")
    p_mapping.add_argument("--mapping", action="store_true", default=False, help="use mapping to get alignment")
    p_mapping.add_argument("--mapq", type=int, default=1, help="mapping quality cutoff, default 1")
    p_mapping.add_argument("--identity", type=float, default=0.0, help="identity cutoff, default 0.0")
    p_mapping.add_argument("--coverage_ratio", type=float, default=0.50, help="coverage percent, default 0.50")
    p_mapping.add_argument("--best_n", "-n", type=int, default=1, help="best_n arg in mappy, default 1")

    parser.add_argument("--nproc", "-p", type=int, default=10, help="number of processes, default 10")
    parser.add_argument("--nproc_gpu", type=int, default=2, help="number of processes to use gpu, default 2")

    args = parser.parse_args()
    display_args(args)
    call_mods(args)

if __name__ == "__main__":
    sys.exit(main())