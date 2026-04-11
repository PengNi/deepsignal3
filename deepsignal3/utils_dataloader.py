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

import pod5
import pyslow5

from .utils.process_utils import get_logger
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import normalize_signals
from .utils.process_utils import base2code_dna
from .utils import bam_reader

LOGGER = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal ↔ base alignment utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    Map query positions to reference positions via CIGAR.
    Returns an array of length seq_len+1.
      -1  → insertion into ref
      -2  → deletion / invalid
    """
    fill_invalid = -2
    q_to_r_poss = np.full(seq_len + 1, fill_invalid, dtype=np.int32)
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar_tuple if strand == 1 else r_cigar_tuple[::-1]
    for op, op_len in cigar_ops:
        if op == 1:                          # insertion
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = -1
            curr_q_pos += op_len
        elif op in (2, 3):                   # deletion / skip
            curr_r_pos += op_len
        elif op in (0, 7, 8):               # match / seq-match / seq-mismatch
            for off in range(op_len):
                q_to_r_poss[curr_q_pos + off] = curr_r_pos + off
            curr_q_pos += op_len
            curr_r_pos += op_len
        # op == 6 (padding) – ignore
    q_to_r_poss[curr_q_pos] = curr_r_pos
    if q_to_r_poss[-1] == fill_invalid:
        raise ValueError(
            f"Invalid CIGAR: ref_len={seq_len}, cigar implied {curr_r_pos}"
        )
    return q_to_r_poss


def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """
    Group raw signals per base using the move table (Python-loop version).
    Used by process_data_bilstm to obtain variable-length per-base signals
    for mean / std / len computation.
    """
    assert movetable[0] == 1
    assert len(trimed_signals) >= len(movetable) * stride
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    signal_group = []
    for i in range(len(move_pos) - 1):
        s, e = move_pos[i], move_pos[i + 1]
        signal_group.append(trimed_signals[s * stride: e * stride].tolist())
    assert len(signal_group) == int(np.sum(movetable))
    return signal_group



def build_signal_rect_from_movetable(trimed_signals, movetable, stride, signals_len=16):
    """
    Vectorised: build (num_events, signals_len) rect array from move table.
    NaN-padded for short events; downsampled for long events.
    Used by process_data_fast (MTM) and process_data_bilstm (BiLSTM).
    """
    move_idx = np.flatnonzero(movetable == 1)
    move_idx = np.append(move_idx, len(movetable))

    starts = move_idx[:-1] * stride
    ends   = move_idx[1:]  * stride
    N      = len(starts)

    out = np.full((N, signals_len), np.nan, dtype=np.float32)
    for i in range(N):
        s, e = starts[i], ends[i]
        sig = trimed_signals[s:e]
        L   = e - s
        if L == 0:
            continue
        if L <= signals_len:
            pad_left = (signals_len - L) // 2
            out[i, pad_left: pad_left + L] = sig
        else:
            idx = np.linspace(0, L - 1, signals_len).astype(np.int32)
            out[i] = sig[idx]
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

    out = []
    for i, loc in enumerate(tsite_locs):
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

        # proximity tag: 0 = isolated, 1 = within 10 bp of another CpG
        tag = 0
        if i > 0 and (loc - tsite_locs[i - 1]) <= 10:
            tag = 1
        elif i < len(tsite_locs) - 1 and (tsite_locs[i + 1] - loc) <= 10:
            tag = 1

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

    def _flush_buffer(qid):
        if buffers[qid]:
            queues[qid].put(buffers[qid])
            buffers[qid] = []

    def _handle_read(signal, read_name):
        try:
            for seq_read in bam_index.get_alignments(read_name):
                feats = process_fn(signal, seq_read, motif_seqs, positions, args)
                for f in feats:
                    qid = random.randint(0, num_workers - 1)
                    buffers[qid].append(f)
                    if len(buffers[qid]) >= BUF_SIZE:
                        _flush_buffer(qid)
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
