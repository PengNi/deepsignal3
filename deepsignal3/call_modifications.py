"""
call_modifications.py
Unified inference entry point: inference_ultra()
Supports: modelMTM / ModelBiLSTM; pod5 / slow5 / fast5 / tsv input; GPU / CPU
"""

from __future__ import absolute_import

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from .models import ModelBiLSTM, modelMTM
from .utils.process_utils import (
    base2code_dna, code2base_dna, str2bool, display_args,
    get_motif_seqs, get_files, read_position_file,
    detect_file_type, get_logger,
)
from .utils_dataloader import producer

LOGGER = get_logger(__name__)
os.environ["MKL_THREADING_LAYER"] = "GNU"


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model_mtm(args, device):
    """Build and load a modelMTM checkpoint onto device."""
    model = modelMTM(
        num_chn=args.mtm_num_base_features + args.n_embed,
        d_static=args.mtm_d_static,
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

    checkpoint = torch.load(args.model_path, map_location="cpu")

    # Clean torch.compile / DDP key prefixes
    clean = {}
    for k, v in checkpoint.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        clean[k] = v

    model.load_state_dict(clean, strict=True)
    model = model.to(device)
    model.eval()

    if getattr(args, "use_compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            LOGGER.warning(f"torch.compile failed, falling back to eager: {e}")

    return model


def load_model_bilstm(args, device):
    """Build and load a ModelBiLSTM checkpoint onto device."""
    model = ModelBiLSTM(
        seq_len=args.seq_len,
        signal_len=args.signal_len,
        num_layers1=args.layernum1,
        num_layers2=args.layernum2,
        num_classes=args.class_num,
        dropout_rate=args.dropout_rate,
        hidden_size=args.hid_rnn,
        vocab_size=args.n_vocab,
        embedding_size=args.n_embed,
        is_base=str2bool(args.is_base),
        is_signallen=str2bool(args.is_signallen),
        is_trace=str2bool(args.is_trace),
        module="both_bilstm",
    )

    checkpoint = torch.load(args.model_path, map_location="cpu")

    # Clean torch.compile / DDP key prefixes
    clean = {}
    for k, v in checkpoint.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        clean[k] = v

    model.load_state_dict(clean, strict=True)
    model = model.to(device)
    model.eval()

    if getattr(args, "use_compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            LOGGER.warning(f"torch.compile failed, falling back to eager: {e}")

    return model


# ─────────────────────────────────────────────
# Batch inference helpers
# ─────────────────────────────────────────────

def _run_batch_mtm(batch_info, batch_k, batch_s, batch_t, args, device, model):
    """
    Run one forward pass for modelMTM.
    Input arrays come from process_data_fast():
        batch_k : list of np.int64   (seq_len,)
        batch_s : list of np.float32 (seq_len, signal_len)
        batch_t : list of int        (proximity tag)
    """
    k_np = np.stack(batch_k)               # (B, L)   – one allocation
    s_np = np.stack(batch_s)               # (B, L, S)
    B, L, S = s_np.shape

    kmers   = torch.from_numpy(k_np).to(device, non_blocking=True)
    signals = torch.from_numpy(s_np.reshape(B, L * S, 1)).to(device, non_blocking=True)
    tags    = torch.tensor(batch_t, dtype=torch.long, device=device)

    # expand kmer codes to match signal time axis
    kmer_expand = kmers.unsqueeze(2).expand(-1, -1, S).reshape(B, -1)  # (B, L*S)

    x_mask = torch.isnan(signals)  # (B, L*S, 1)
    false_mask = torch.zeros(
        (*x_mask.shape[:-1], args.n_embed), device=device, dtype=torch.bool
    )
    x_mask = torch.cat([x_mask, false_mask], dim=-1)  # (B, L*S, 1+n_embed)

    tpos     = torch.arange(L * S, device=device).unsqueeze(0).expand(B, -1)  # (B, L*S)
    x_static = tags.unsqueeze(-1)                                              # (B, 1)

    with torch.no_grad():
        logits = model(signals, kmer_expand, x_mask, tpos, x_static)
        probs  = torch.softmax(logits, dim=-1)
        pred   = torch.argmax(logits, dim=1)

    probs_np = probs.cpu().numpy()
    pred_np  = pred.cpu().numpy()
    kmers_np = kmers.cpu().numpy()

    out_lines = []
    for i in range(len(batch_info)):
        p0 = round(float(probs_np[i][0]), 6)
        p1 = round(float(probs_np[i][1]), 6)
        kseq   = "".join(code2base_dna[int(x)] for x in kmers_np[i])
        c      = len(kseq) // 2
        kmer5  = kseq[max(c - 2, 0):c + 3]
        out_lines.append("\t".join([batch_info[i], str(p0), str(p1),
                                    str(int(pred_np[i])), kmer5]))

    if device.type == "cuda":
        del kmers, signals, tags, logits, probs, pred, kmer_expand, x_mask, false_mask
        torch.cuda.empty_cache()

    return out_lines


def _run_batch_bilstm(batch_info, batch_k, batch_means, batch_stds,
                      batch_lens, batch_s, device, model):
    """
    Run one forward pass for ModelBiLSTM.
    Input tensors come from process_data_bilstm():
        batch_k     : list of np.int64   (seq_len,)
        batch_means : list of np.float32 (seq_len,)
        batch_stds  : list of np.float32 (seq_len,)
        batch_lens  : list of np.int32   (seq_len,)
        batch_s     : list of np.float32 (seq_len, signal_len)
    """
    kmers   = torch.stack(batch_k).to(device, non_blocking=True)      # (B, L)
    means   = torch.stack(batch_means).to(device, non_blocking=True)  # (B, L)
    stds    = torch.stack(batch_stds).to(device, non_blocking=True)   # (B, L)
    lens    = torch.stack(batch_lens).to(device, non_blocking=True)   # (B, L)
    signals = torch.stack(batch_s).to(device, non_blocking=True)      # (B, L, S)

    with torch.no_grad():
        # ModelBiLSTM.forward returns (logits, softmax_probs)
        _, probs = model(kmers.long(), means, stds, lens, signals)
        pred = torch.argmax(probs, dim=1)

    probs_np = probs.cpu().numpy()
    pred_np  = pred.cpu().numpy()
    kmers_np = kmers.cpu().numpy()

    out_lines = []
    for i in range(len(batch_info)):
        p0 = round(float(probs_np[i][0]), 6)
        p1 = round(float(probs_np[i][1]), 6)
        kseq  = "".join(code2base_dna[int(x)] for x in kmers_np[i])
        c     = len(kseq) // 2
        kmer5 = kseq[max(c - 2, 0):c + 3]
        out_lines.append("\t".join([batch_info[i], str(p0), str(p1),
                                    str(int(pred_np[i])), kmer5]))

    if device.type == "cuda":
        del kmers, means, stds, lens, signals, probs, pred
        torch.cuda.empty_cache()

    return out_lines


# ─────────────────────────────────────────────
# Unified model worker  (GPU rank or CPU)
# ─────────────────────────────────────────────

def model_worker(rank, device, queue, pred_q, args, nproc_io):
    """
    Consume feature items from queue, run inference, push result lines to pred_q.
    Works for both GPU (device=cuda:N) and CPU (device=cpu).
    Works for both modelMTM and ModelBiLSTM (controlled by args.model_class).
    """
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
    else:
        # Avoid CPU thread contention with multiprocessing workers
        torch.set_num_threads(1)

    is_mtm = (args.model_class == "mtm")

    if is_mtm:
        model = load_model_mtm(args, device)
    else:
        model = load_model_bilstm(args, device)
    model.eval()

    # ── batch buffers ──────────────────────────────
    batch_info = []
    batch_k, batch_s = [], []
    batch_t = []                                        # MTM only: proximity tag
    batch_means, batch_stds, batch_lens = [], [], []    # BiLSTM only
    end_count = 0

    def flush():
        if not batch_info:
            return
        if is_mtm:
            lines = _run_batch_mtm(
                batch_info, batch_k, batch_s, batch_t,
                args, device, model,
            )
        else:
            lines = _run_batch_bilstm(
                batch_info, batch_k, batch_means, batch_stds,
                batch_lens, batch_s, device, model,
            )
        if lines:
            pred_q.put(lines)
        batch_info.clear(); batch_k.clear(); batch_s.clear(); batch_t.clear()
        batch_means.clear(); batch_stds.clear(); batch_lens.clear()

    # ── main loop ──────────────────────────────────
    while True:
        item = queue.get()

        if item is None:
            end_count += 1
            if end_count == nproc_io:
                break
            continue

        items = item if isinstance(item, list) else [item]

        for sub in items:
            if is_mtm:
                # sub = (sampleinfo, k_seq, k_signals_rect, label, tag)
                # label (sub[3]) discarded – not needed for inference
                batch_info.append(sub[0])
                batch_k.append(np.asarray(sub[1], dtype=np.int64))
                batch_s.append(np.asarray(sub[2], dtype=np.float32))
                batch_t.append(sub[4])
            else:
                # sub = (sampleinfo, k_seq, means, stds, lens, k_signals_rect, label)
                # label (sub[6]) discarded – not needed for inference
                batch_info.append(sub[0])
                batch_k.append(torch.from_numpy(np.asarray(sub[1], dtype=np.int64)))
                batch_means.append(torch.from_numpy(np.asarray(sub[2], dtype=np.float32)))
                batch_stds.append(torch.from_numpy(np.asarray(sub[3], dtype=np.float32)))
                batch_lens.append(torch.from_numpy(np.asarray(sub[4], dtype=np.float32)))
                batch_s.append(torch.from_numpy(np.asarray(sub[5], dtype=np.float32)))

            if len(batch_info) >= args.batch_size:
                flush()

    # flush tail batch
    flush()
    print(f"[Worker-{rank}({device})] done", flush=True)


# ─────────────────────────────────────────────
# TSV producer  (reads pre-extracted feature file)
# ─────────────────────────────────────────────

def tsv_producer(tsv_file, queues, args):
    """
    Read a pre-extracted feature TSV (or .gz) and distribute items to model workers.
    Supports both MTM and BiLSTM output formats automatically.

    TSV columns (12):
        chrom, pos, strand, loc_in_strand, readname, read_loc,
        k_mer, signal_means, signal_stds, signal_lens, k_signals_rect, label
    """
    import random
    import gzip

    is_mtm = (args.model_class == "mtm")
    n_workers = len(queues)
    BUF_SIZE = 128
    buffers = [[] for _ in range(n_workers)]

    open_fn = gzip.open if tsv_file.endswith(".gz") else open
    with open_fn(tsv_file, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            words = line.split("\t")
            if len(words) < 12:
                continue

            sampleinfo = "\t".join(words[:6])
            k_mer = words[6]
            k_seq = np.fromiter(
                (base2code_dna[x] for x in k_mer),
                dtype=np.int64, count=len(k_mer),
            )
            k_signals = np.array(
                [[float(y) for y in x.split(",")] for x in words[10].split(";")],
                dtype=np.float32,
            )
            label = int(words[11])

            if is_mtm:
                # tag not stored in TSV → default 1 (no proximity filtering)
                item = (sampleinfo, k_seq, k_signals, label, 1)
            else:
                means = np.array([float(x) for x in words[7].split(",")], dtype=np.float32)
                stds  = np.array([float(x) for x in words[8].split(",")], dtype=np.float32)
                lens  = np.array([int(x)   for x in words[9].split(",")], dtype=np.int32)
                item  = (sampleinfo, k_seq, means, stds, lens, k_signals, label)

            qid = random.randint(0, n_workers - 1)
            buffers[qid].append(item)
            if len(buffers[qid]) >= BUF_SIZE:
                queues[qid].put(buffers[qid])
                buffers[qid] = []

    for qid in range(n_workers):
        if buffers[qid]:
            queues[qid].put(buffers[qid])

    print("[TSV-Producer] done", flush=True)


# ─────────────────────────────────────────────
# Writer process
# ─────────────────────────────────────────────

def writer(out_file, pred_q):
    with open(out_file, "w") as f:
        while True:
            item = pred_q.get()
            if item == "kill":
                break
            f.write("\n".join(item) + "\n")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def inference_ultra(args):
    """
    Unified inference pipeline.
    - Input:  pod5 / slow5 / fast5 directory, or pre-extracted TSV file
    - Model:  modelMTM (args.model_class='mtm') or ModelBiLSTM (args.model_class='bilstm')
    - Device: all available GPUs; falls back to CPU when none found or --use_cpu set
    """
    mp.set_start_method("spawn", force=True)

    # ── device selection ──────────────────────────
    num_gpu = torch.cuda.device_count()
    use_gpu = num_gpu > 0 and not getattr(args, "use_cpu", False)

    if use_gpu:
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpu)]
        LOGGER.info(f"Using {num_gpu} GPU(s): {devices}")
    else:
        devices = [torch.device("cpu")]
        LOGGER.info("No GPU available or --use_cpu set. Running on CPU.")

    n_workers = len(devices)
    nproc_io  = max(1, args.nproc)

    queues = [mp.Queue(256) for _ in range(n_workers)]
    pred_q = mp.Queue(512)

    # ── detect input type ─────────────────────────
    input_path = args.input_path
    is_tsv = os.path.isfile(input_path) and (
        input_path.endswith(".tsv") or input_path.endswith(".gz")
    )

    producers   = []
    _nproc_io_actual = nproc_io  # used for end-signal count

    if is_tsv:
        LOGGER.info(f"TSV input detected: {input_path}")
        p = mp.Process(
            target=tsv_producer,
            args=(input_path, queues, args),
        )
        p.start()
        producers.append(p)
        _nproc_io_actual = 1  # single TSV reader

    else:
        file_type = detect_file_type(input_path, True)
        files = get_files(input_path, True, file_type)
        LOGGER.info(f"Signal input: {len(files)} {file_type} files, "
                    f"{nproc_io} IO worker(s)")

        motif_seqs = get_motif_seqs(args.motifs, True)
        positions  = read_position_file(args.positions) if args.positions else None

        for i in range(nproc_io):
            p = mp.Process(
                target=producer,
                args=(i, files, queues, args, motif_seqs, positions,
                      file_type, n_workers, nproc_io),
            )
            p.start()
            producers.append(p)

    # ── model workers ─────────────────────────────
    workers = []
    for rank, device in enumerate(devices):
        p = mp.Process(
            target=model_worker,
            args=(rank, device, queues[rank], pred_q, args, _nproc_io_actual),
        )
        p.start()
        workers.append(p)

    # ── writer ────────────────────────────────────
    p_writer = mp.Process(target=writer, args=(args.result_file, pred_q))
    p_writer.start()

    # ── wait for producers ────────────────────────
    for p in producers:
        p.join()
    LOGGER.info("All producers done.")

    # send end signals  (one None per IO worker per model worker queue)
    for q in queues:
        for _ in range(_nproc_io_actual):
            q.put(None)

    for p in workers:
        p.join()

    pred_q.put("kill")
    p_writer.join()

    LOGGER.info("ALL DONE")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "deepsignal3 call_mods – unified inference (modelMTM / BiLSTM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input ──────────────────────────────────────
    p_in = parser.add_argument_group("INPUT")
    p_in.add_argument("--input_path", "-i", required=True,
                      help="Signal directory (pod5/slow5/fast5) or pre-extracted TSV file")
    p_in.add_argument("--bam", type=str, default=None,
                      help="BAM file (required for pod5/slow5/fast5 input)")
    p_in.add_argument("--recursively", "-r", type=str, default="yes",
                      help="Search files recursively")

    # ── Output ─────────────────────────────────────
    p_out = parser.add_argument_group("OUTPUT")
    p_out.add_argument("--result_file", "-o", required=True,
                       help="Path to write prediction results")

    # ── Model selection ────────────────────────────
    p_model = parser.add_argument_group("MODEL")
    p_model.add_argument("--model_path", "-m", required=True,
                         help="Path to trained model checkpoint (.ckpt)")
    p_model.add_argument("--model_class", type=str, default="mtm",
                         choices=["mtm", "bilstm"],
                         help="Model architecture: 'mtm' (modelMTM) or 'bilstm' (ModelBiLSTM)")
    p_model.add_argument("--use_compile", type=str2bool, default="no",
                         help="Enable torch.compile (PyTorch >= 2.0, may speed up GPU inference)")
    p_model.add_argument("--use_cpu", action="store_true", default=False,
                         help="Force CPU inference even when GPUs are available")

    # ── Shared hyper-params (both models) ──────────
    p_hp = parser.add_argument_group("MODEL_HYPER")
    p_hp.add_argument("--seq_len",      type=int,   default=21,
                      help="K-mer length (must be odd)")
    p_hp.add_argument("--signal_len",   type=int,   default=15,
                      help="Signals per base")
    p_hp.add_argument("--class_num",    type=int,   default=2)
    p_hp.add_argument("--dropout_rate", type=float, default=0.0)
    p_hp.add_argument("--n_vocab",      type=int,   default=16,
                      help="Base vocabulary size")
    p_hp.add_argument("--n_embed",      type=int,   default=4,
                      help="Base embedding dimension")

    # ── BiLSTM-specific ────────────────────────────
    p_lstm = parser.add_argument_group("BILSTM_HYPER")
    p_lstm.add_argument("--hid_rnn",      type=int, default=256,
                        help="Hidden size for BiLSTM")
    p_lstm.add_argument("--layernum1",    type=int, default=3)
    p_lstm.add_argument("--layernum2",    type=int, default=1)
    p_lstm.add_argument("--is_base",      type=str, default="yes")
    p_lstm.add_argument("--is_signallen", type=str, default="yes")
    p_lstm.add_argument("--is_trace",     type=str, default="no")

    # ── MTM-specific ───────────────────────────────
    p_mtm = parser.add_argument_group("MTM_HYPER")
    p_mtm.add_argument("--mtm_num_base_features", type=int,   default=1)
    p_mtm.add_argument("--mtm_hid_rnn",            type=int,   default=128,
                        help="d_model (hidden size) for MTM")
    p_mtm.add_argument("--mtm_d_static",          type=int,   default=1)
    p_mtm.add_argument("--mtm_ratios",    nargs="+", type=int, default=[2, 2, 2, 2])
    p_mtm.add_argument("--mtm_r_hid",             type=int,   default=4)
    p_mtm.add_argument("--mtm_norm_first",  type=str2bool,    default="True")
    p_mtm.add_argument("--mtm_down_mode",   type=str,         default="concat",
                        choices=["concat", "avg", "max"])
    p_mtm.add_argument("--mtm_temporal_depth", type=int,      default=2)

    # ── Extraction / mapping ───────────────────────
    p_ext = parser.add_argument_group("EXTRACTION")
    p_ext.add_argument("--motifs",           type=str,   default="CG")
    p_ext.add_argument("--mod_loc",          type=int,   default=0)
    p_ext.add_argument("--methy_label",      type=int,   default=1, choices=[0, 1])
    p_ext.add_argument("--normalize_method", type=str,   default="mad",
                        choices=["mad", "zscore"])
    p_ext.add_argument("--mapq",             type=int,   default=1)
    p_ext.add_argument("--coverage_ratio",   type=float, default=0.5)
    p_ext.add_argument("--identity",         type=float, default=0.0)
    p_ext.add_argument("--positions",        type=str,   default=None,
                        help="Position filter file")
    p_ext.add_argument("--rna",              action="store_true", default=False)
    p_ext.add_argument("--plant",            action="store_true", default=False,
                        help="Plant mode: proximity tag counts any C within "
                             "±10 bp (motif-agnostic). Default (human mode): "
                             "only same-motif sites are counted.")

    # ── Performance ────────────────────────────────
    p_perf = parser.add_argument_group("PERFORMANCE")
    p_perf.add_argument("--batch_size", "-b", type=int, default=500)
    p_perf.add_argument("--nproc",      "-p", type=int, default=10,
                         help="Number of IO producer processes")

    args = parser.parse_args()
    display_args(args)
    inference_ultra(args)


if __name__ == "__main__":
    sys.exit(main())
