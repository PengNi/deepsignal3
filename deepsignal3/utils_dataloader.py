"""
utils_dataloader.py
Core data processing functions and IO producer for inference.

Active components:
  - build_signal_rect_from_movetable  : fast vectorised signal-to-base mapping
  - get_q2tloc_from_cigar             : CIGAR → query-to-ref position mapping
  - _group_signals_by_movetable_v2    : variable-length signal grouping (BiLSTM)
  - _get_signals_rect                 : pad/trim signal windows to fixed length (BiLSTM)
  - process_data_fast                 : feature extraction for modelMTM (no mean/std/len)
  - process_data_bilstm               : feature extraction for ModelBiLSTM (with mean/std/len)
  - producer                          : multi-process IO worker (pod5 / slow5)
"""

import random
import numpy as np
from numba import jit

import pod5
import pyslow5

from .utils.process_utils import get_logger
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import compute_proximity_tag
from .utils.process_utils import normalize_signals
from .utils.process_utils import base2code_dna
from .utils import bam_reader

LOGGER = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal ↔ base alignment utilities
# ─────────────────────────────────────────────────────────────────────────────

@jit(nopython=True, cache=True)
def _q2tloc_jit(cigar_ops, cigar_lens, seq_len, forward):
    q_to_r_poss = np.full(seq_len + 1, np.int32(-2), dtype=np.int32)
    curr_r_pos = np.int32(0)
    curr_q_pos = np.int32(0)
    n = len(cigar_ops)
    for ii in range(n):
        i = ii if forward else (n - 1 - ii)
        op     = cigar_ops[i]
        op_len = cigar_lens[i]
        if op == 1:                         # insertion
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = np.int32(-1)
            curr_q_pos += op_len
        elif op == 2 or op == 3:            # deletion / skip
            curr_r_pos += op_len
        elif op == 0 or op == 7 or op == 8: # match / seq-match / seq-mismatch
            for off in range(op_len):
                q_to_r_poss[curr_q_pos + off] = curr_r_pos + off
            curr_q_pos += op_len
            curr_r_pos += op_len
    q_to_r_poss[curr_q_pos] = curr_r_pos
    return q_to_r_poss


def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    Map query positions to reference positions via CIGAR.
    Returns an array of length seq_len+1.
      -1  → insertion into ref
      -2  → deletion / invalid
    """
    ops  = np.array([op  for op, _  in r_cigar_tuple], dtype=np.int32)
    lens = np.array([ln  for _,  ln in r_cigar_tuple], dtype=np.int32)
    q_to_r_poss = _q2tloc_jit(ops, lens, seq_len, strand == 1)
    if q_to_r_poss[-1] == -2:
        raise ValueError(
            f"Invalid CIGAR: ref_len={seq_len}, cigar did not cover full query"
        )
    return q_to_r_poss


def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """
    Group raw signals per base using the move table (Python-loop version).
    Used by process_data_bilstm to obtain variable-length per-base signals
    for mean / std / len computation.
    """
    if movetable[0] != 1:
        raise ValueError(
            f"move table must start with 1 (a move), got {movetable[0]}"
        )
    if len(trimed_signals) < len(movetable) * stride:
        raise ValueError(
            f"trimmed signal length ({len(trimed_signals)}) is shorter than "
            f"expected ({len(movetable)} moves × stride {stride} = {len(movetable) * stride})"
        )
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    signal_group = []
    for i in range(len(move_pos) - 1):
        s, e = move_pos[i], move_pos[i + 1]
        signal_group.append(trimed_signals[s * stride: e * stride].tolist())
    assert len(signal_group) == int(np.sum(movetable))
    return signal_group



@jit(nopython=True, cache=True)
def _build_signal_rect_jit(sig, starts, ends, signals_len):
    """JIT kernel: fills rect with 0.0 and marks valid positions in a bool mask."""
    N   = len(starts)
    out   = np.zeros((N, signals_len), dtype=np.float32)
    valid = np.ones((N, signals_len),  dtype=np.bool_)
    for i in range(N):
        s = starts[i]
        e = ends[i]
        L = e - s
        if L == 0:
            for j in range(signals_len):
                valid[i, j] = False
            continue
        if L <= signals_len:
            pad_left = (signals_len - L) // 2
            for j in range(pad_left):
                valid[i, j] = False
            for j in range(L):
                out[i, pad_left + j] = sig[s + j]
            for j in range(pad_left + L, signals_len):
                valid[i, j] = False
        else:
            # downsample: evenly-spaced indices across [s, e)
            for j in range(signals_len):
                idx = s + int(j * (L - 1) / (signals_len - 1))
                out[i, j] = sig[idx]
    return out, valid


def build_signal_rect_from_movetable(trimed_signals, movetable, stride, signals_len=16):
    """
    Build (num_events, signals_len) rect array from move table.
    NaN-padded for short events; downsampled for long events.
    Used by process_data_fast (MTM) and process_data_bilstm (BiLSTM).
    """
    move_idx = np.flatnonzero(movetable == 1)
    move_idx = np.append(move_idx, len(movetable))
    starts = (move_idx[:-1] * stride).astype(np.int64)
    ends   = (move_idx[1:]  * stride).astype(np.int64)

    sig = np.ascontiguousarray(trimed_signals, dtype=np.float32)
    out, valid = _build_signal_rect_jit(sig, starts, ends, signals_len)
    out[~valid] = np.nan   # restore NaN padding for MTM mask (torch.isnan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction  (per-read, called from producer)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bam_read(signal, seq_read, args):
    """
    Shared pre-processing: trim, normalise, build rect signal matrix.
    Returns (norm_signal, signal_rect, movetable, stride) or None on failure.
    """
    seq = seq_read.get_forward_sequence()
    if seq is None:
        return None

    read_dict = dict(seq_read.tags)
    if "mv" not in read_dict:
        return None

    mv     = np.asarray(read_dict["mv"], dtype=np.int32)
    stride = int(mv[0])
    movetable = mv[1:]

    num_trimmed = read_dict["ts"]
    if seq_read.has_tag("sp"):
        num_trimmed += seq_read.get_tag("sp")

    sig_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
    norm_signal  = normalize_signals(sig_trimmed, args.normalize_method)
    signal_rect  = build_signal_rect_from_movetable(norm_signal, movetable, stride, args.signal_len)

    return seq, norm_signal, signal_rect, movetable, stride


def _get_ref_coords(seq_read, seq):
    """
    Compute strand, ref coords, and q→r mapping for a mapped read.
    Returns dict with keys: strand, ref_name, ref_start, ref_end,
    seq_start, seq_end, q_to_r_poss.
    Returns None if read is unmapped.
    """
    if seq_read.is_unmapped:
        return None

    strand     = "-" if seq_read.is_reverse else "+"
    strand_code = -1 if seq_read.is_reverse else 1
    ref_name   = seq_read.reference_name or "."
    ref_start  = seq_read.reference_start
    ref_end    = seq_read.reference_end

    qa_start = seq_read.query_alignment_start
    qa_end   = seq_read.query_alignment_end

    if seq_read.is_reverse:
        seq_start = len(seq) - qa_end
        seq_end   = len(seq) - qa_start
    else:
        seq_start, seq_end = qa_start, qa_end

    q_to_r_poss = get_q2tloc_from_cigar(
        seq_read.cigartuples, strand_code, seq_end - seq_start
    )
    return dict(
        strand=strand, ref_name=ref_name,
        ref_start=ref_start, ref_end=ref_end,
        seq_start=seq_start, seq_end=seq_end,
        q_to_r_poss=q_to_r_poss,
    )


def process_data_fast(signal, seq_read, motif_seqs, positions, args):
    """
    Extract features for modelMTM inference.
    Output per site: (sampleinfo, k_seq[int64], k_signals_rect[float32], label, tag)
    No mean/std/len – fastest path.

    args.plant (bool):
        True  → tag counts any C within ±10 bp (plant / multi-motif mode)
        False → tag counts only same-motif sites within ±10 bp (human / CG mode)
    """
    parsed = _parse_bam_read(signal, seq_read, args)
    if parsed is None:
        return []
    seq, _, signal_rect, _, _ = parsed

    if seq_read.mapping_quality < args.mapq:
        return []

    tsite_locs = get_refloc_of_methysite_in_motif(seq, motif_seqs, args.mod_loc)
    if not tsite_locs:
        return []

    num_bases = (args.seq_len - 1) // 2
    coords    = _get_ref_coords(seq_read, seq)

    # coverage filter (mapped reads only)
    if not seq_read.is_unmapped:
        qa_start = seq_read.query_alignment_start
        qa_end   = seq_read.query_alignment_end
        if (qa_end - qa_start) / seq_read.query_length < args.coverage_ratio:
            return []

    strand   = coords["strand"]   if coords else "."
    ref_name = coords["ref_name"] if coords else "."

    chrom_args = getattr(args, "chrom", None) or []
    _excl = {c[2:] for c in chrom_args if c.startswith("no")}
    _incl = {c for c in chrom_args if not c.startswith("no")}
    if (_excl and ref_name in _excl) or (_incl and ref_name not in _incl):
        return []

    # Pre-compute tag_locs once per read
    plant = getattr(args, "plant", False)
    if plant:
        tag_locs = [i for i, b in enumerate(seq) if b == "C"]
    else:
        tag_locs = tsite_locs  # already sorted

    out = []
    for loc in tsite_locs:
        if not (num_bases <= loc < len(seq) - num_bases):
            continue

        ref_pos = -1
        if coords:
            s, e = coords["seq_start"], coords["seq_end"]
            if not (s <= loc < e):
                continue
            rpos = coords["q_to_r_poss"][loc - s]
            if rpos == -1:
                continue
            ref_pos = (
                coords["ref_end"] - 1 - rpos if strand == "-"
                else coords["ref_start"] + rpos
            )

        if positions is not None:
            if f"{ref_name}\t{ref_pos}\t{strand}" not in positions:
                continue

        tag = compute_proximity_tag(loc, tag_locs, window=10)

        k_mer = seq[loc - num_bases: loc + num_bases + 1]
        k_seq = np.fromiter(
            (base2code_dna[x] for x in k_mer),
            dtype=np.int64, count=args.seq_len,
        )
        k_signals = signal_rect[loc - num_bases: loc + num_bases + 1]
        sampleinfo = f"{ref_name}\t{ref_pos}\t{strand}\t.\t{seq_read.query_name}\t."

        out.append((sampleinfo, k_seq, k_signals, args.methy_label, tag))

    return out


def process_data_bilstm(signal, seq_read, motif_seqs, positions, args):
    """
    Extract features for ModelBiLSTM inference.
    Output per site:
        (sampleinfo, k_seq[int64], means[float32], stds[float32],
         lens[int32], k_signals_rect[float32], label)
    Computes per-base mean/std/len via variable-length signal grouping.

    args.plant (bool): same semantics as process_data_fast.
    """
    parsed = _parse_bam_read(signal, seq_read, args)
    if parsed is None:
        return []
    seq, norm_signal, signal_rect, movetable, stride = parsed

    if seq_read.mapping_quality < args.mapq:
        return []

    # variable-length grouping for mean/std/len
    signal_group = _group_signals_by_movetable_v2(norm_signal, movetable, stride)

    tsite_locs = get_refloc_of_methysite_in_motif(seq, motif_seqs, args.mod_loc)
    if not tsite_locs:
        return []

    num_bases = (args.seq_len - 1) // 2
    coords    = _get_ref_coords(seq_read, seq)

    if not seq_read.is_unmapped:
        qa_start = seq_read.query_alignment_start
        qa_end   = seq_read.query_alignment_end
        if (qa_end - qa_start) / seq_read.query_length < args.coverage_ratio:
            return []

    strand   = coords["strand"]   if coords else "."
    ref_name = coords["ref_name"] if coords else "."

    chrom_args = getattr(args, "chrom", None) or []
    _excl = {c[2:] for c in chrom_args if c.startswith("no")}
    _incl = {c for c in chrom_args if not c.startswith("no")}
    if (_excl and ref_name in _excl) or (_incl and ref_name not in _incl):
        return []

    # Pre-compute tag_locs once per read (BiLSTM currently doesn't use tag,
    # but keeping the logic symmetric for future use)
    plant = getattr(args, "plant", False)
    if plant:
        tag_locs = [i for i, b in enumerate(seq) if b == "C"]
    else:
        tag_locs = tsite_locs  # already sorted

    out = []
    for loc in tsite_locs:
        if not (num_bases <= loc < len(seq) - num_bases):
            continue

        ref_pos = -1
        if coords:
            s, e = coords["seq_start"], coords["seq_end"]
            if not (s <= loc < e):
                continue
            rpos = coords["q_to_r_poss"][loc - s]
            if rpos == -1:
                continue
            ref_pos = (
                coords["ref_end"] - 1 - rpos if strand == "-"
                else coords["ref_start"] + rpos
            )

        if positions is not None:
            if f"{ref_name}\t{ref_pos}\t{strand}" not in positions:
                continue

        k_mer    = seq[loc - num_bases: loc + num_bases + 1]
        k_seq    = np.fromiter(
            (base2code_dna[x] for x in k_mer),
            dtype=np.int64, count=args.seq_len,
        )
        k_sigs_v = signal_group[loc - num_bases: loc + num_bases + 1]
        means    = np.array([np.mean(x) for x in k_sigs_v], dtype=np.float32)
        stds     = np.array([np.std(x)  for x in k_sigs_v], dtype=np.float32)
        lens     = np.array([len(x)     for x in k_sigs_v], dtype=np.int32)
        k_signals = signal_rect[loc - num_bases: loc + num_bases + 1]
        sampleinfo = f"{ref_name}\t{ref_pos}\t{strand}\t.\t{seq_read.query_name}\t."

        out.append((sampleinfo, k_seq, means, stds, lens, k_signals, args.methy_label))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# IO Producer  (one process per worker_id shard)
# ─────────────────────────────────────────────────────────────────────────────

def producer(worker_id, files, queues, args, motif_seqs, positions,
             file_type, num_workers, nproc_io):
    """
    Read signal files (pod5 / slow5), extract features, distribute to model workers.

    Each worker handles files[worker_id::nproc_io] (round-robin sharding).
    Items are buffered and sent in batches of BUF_SIZE to reduce IPC overhead.

    Supports:
      - pod5  : pod5.Reader
      - slow5 : pyslow5.Open (includes .blow5)
    """
    my_files = files[worker_id::nproc_io]
    print(f"[Producer-{worker_id}] {len(my_files)} {file_type} files", flush=True)

    bam_index = bam_reader.ReadIndexedBam(args.bam)

    # choose processing function based on model class
    is_bilstm = (getattr(args, "model_class", "mtm") == "bilstm")
    process_fn = process_data_bilstm if is_bilstm else process_data_fast

    BUF_SIZE = 128
    buffers  = [[] for _ in range(num_workers)]

    # ── per-producer timing (set DEEPSIGNAL_PROFILE=1 to enable) ──────────
    import os, time as _time
    _profile = os.environ.get("DEEPSIGNAL_PROFILE") == "1"
    _t_norm = _t_proc = _t_bam = 0.0
    _n_reads = 0

    def _flush_buffer(qid):
        if buffers[qid]:
            queues[qid].put(buffers[qid])
            buffers[qid] = []

    def _handle_read(signal, read_name):
        nonlocal _t_norm, _t_proc, _t_bam, _n_reads
        try:
            if _profile:
                _t0 = _time.perf_counter()
            aligns = list(bam_index.get_alignments(read_name))
            if _profile:
                _t_bam += _time.perf_counter() - _t0
            for seq_read in aligns:
                if _profile:
                    _t0 = _time.perf_counter()
                feats = process_fn(signal, seq_read, motif_seqs, positions, args)
                if _profile:
                    _t_proc += _time.perf_counter() - _t0
                for f in feats:
                    qid = random.randint(0, num_workers - 1)
                    buffers[qid].append(f)
                    if len(buffers[qid]) >= BUF_SIZE:
                        _flush_buffer(qid)
            if _profile:
                _n_reads += 1
                if _n_reads % 500 == 0:
                    print(
                        f"[Producer-{worker_id}] {_n_reads} reads | "
                        f"bam_lookup {_t_bam*1e3/500:.2f} ms/r | "
                        f"process_fn {_t_proc*1e3/500:.2f} ms/r",
                        flush=True,
                    )
                    _t_norm = _t_proc = _t_bam = 0.0
        except KeyError:
            pass  # read not in BAM – skip silently

    for file in my_files:
        try:
            if file_type == "pod5":
                with pod5.Reader(file) as reader:
                    for read in reader.reads():
                        _handle_read(read.signal, str(read.read_id))

            elif file_type in ("slow5", "blow5"):
                s5 = pyslow5.Open(file, "r")
                try:
                    for read in s5.seq_reads():
                        _handle_read(read["signal"], read["read_id"])
                finally:
                    s5.close()

            elif file_type == "fast5":
                from .utils import fast5_reader
                is_single = getattr(args, "single", False)
                if is_single:
                    f5 = fast5_reader.SingleFast5(file, is_single=True)
                    try:
                        sig = f5.rescale_signals(f5.get_raw_signal())
                        _handle_read(sig, f5.get_readid())
                    finally:
                        f5.close()
                else:
                    mf = fast5_reader.MultiFast5(file)
                    try:
                        for rname in mf:
                            f5 = fast5_reader.SingleFast5(mf[rname], readname=rname)
                            sig = f5.rescale_signals(f5.get_raw_signal())
                            _handle_read(sig, f5.get_readid())
                    finally:
                        mf.close()

        except Exception as e:
            print(f"[Producer-{worker_id}] error on {file}: {e}", flush=True)

    # flush remaining items
    for qid in range(num_workers):
        _flush_buffer(qid)

    print(f"[Producer-{worker_id}] done", flush=True)
