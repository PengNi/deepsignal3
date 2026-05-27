"""
extract_features_pod5.py
Feature extraction for training/testing data.

Output TSV format (tab-separated):
  chrom, pos, alignstrand, pos_in_strand, readname, read_loc,
  k_mer, signal_means, signal_stds, signal_lens, raw_signals, methy_label

Supports pod5 and slow5/blow5 input.
"""

from __future__ import absolute_import

import sys
import os
import argparse
import time

import numpy as np
import multiprocessing as mp
from multiprocessing import Queue

import pod5
import pyslow5

from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import get_files
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import compute_proximity_tag
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import fill_files_queue
from .utils.process_utils import read_position_file
from .utils.process_utils import write_featurestr
from .utils.process_utils import normalize_signals
from .utils.process_utils import get_logger
from .utils.process_utils import key_sep
from .utils.process_utils import detect_file_type

from .utils import bam_reader

from .utils_dataloader import (
    get_q2tloc_from_cigar,
    build_signal_rect_from_movetable,
    _group_signals_by_movetable_v2,
)

LOGGER = get_logger(__name__)

time_wait = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# TSV serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _features_to_str(features):
    (
        chrom,
        pos,
        alignstrand,
        loc_in_ref,
        readname,
        read_loc,
        k_mer,
        signal_means,
        signal_stds,
        signal_lens,
        k_signals_rect,   # (kmer_len, signals_len) float32 array, 0-padded
        methy_label,
        tag,              # 0/1 — any C within ±10 bp of the target site in read seq
    ) = features
    means_text      = ",".join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text       = ",".join([str(x) for x in np.around(signal_stds,  decimals=6)])
    signal_len_text = ",".join([str(x) for x in signal_lens])
    k_signals_text  = ";".join(
        [",".join([str(round(float(y), 6)) for y in row]) for row in k_signals_rect]
    )
    return "\t".join([
        chrom,
        str(pos),
        alignstrand,
        str(loc_in_ref),
        readname,
        str(read_loc),
        k_mer,
        means_text,
        stds_text,
        signal_len_text,
        k_signals_text,
        str(methy_label),
        str(tag),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Per-read feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def process_data(
    signal,
    seq_read,
    motif_seqs,
    positions,
    kmer_len,
    signals_len,
    mapq,
    coverage_ratio,
    identity,
    methyloc=0,
    methy_label=1,
    norm_method="mad",
    plant=False,
):
    """Extract TSV feature strings from one (signal, seq_read) pair.

    Parameters
    ----------
    plant : bool
        If True  (plant mode) : tag = 1 when any other C exists within ±10 bp.
        If False (human mode) : tag = 1 when any other same-motif site (as
            defined by motif_seqs) exists within ±10 bp.
        Both modes use the same read-sequence-based computation, so training
        and inference are always consistent when the same flag is passed.
    """
    if kmer_len % 2 == 0:
        raise ValueError("kmer_len must be odd")
    num_bases = (kmer_len - 1) // 2
    features_list = []

    if seq_read.mapping_quality < mapq:
        return features_list

    seq = seq_read.get_forward_sequence()
    if seq is None:
        return features_list

    read_dict = dict(seq_read.tags)
    if "mv" not in read_dict:
        return features_list

    mv     = np.asarray(read_dict["mv"], dtype=np.int32)
    stride = int(mv[0])
    mv_table = mv[1:]

    num_trimmed = read_dict["ts"]
    if seq_read.has_tag("sp"):
        num_trimmed += seq_read.get_tag("sp")

    sig_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
    norm_signals = normalize_signals(sig_trimmed, norm_method)

    # vectorised signal → base rect matrix (NaN-padded)
    signal_rect_full = build_signal_rect_from_movetable(norm_signals, mv_table, stride, signals_len)

    # variable-length grouping for mean/std/len
    signal_group = _group_signals_by_movetable_v2(norm_signals, mv_table, stride)

    tsite_locs = get_refloc_of_methysite_in_motif(seq, set(motif_seqs), methyloc)

    # Pre-compute tag reference locs once per read (not per site):
    #   plant=True  → all C positions (motif-agnostic)
    #   plant=False → same-motif positions only (= tsite_locs itself)
    if plant:
        tag_locs = [i for i, b in enumerate(seq) if b == "C"]
    else:
        tag_locs = tsite_locs   # already sorted

    strand      = "."
    ref_name    = "."
    ref_start   = ref_end = 0
    seq_start   = seq_end = 0
    q_to_r_poss = None

    if not seq_read.is_unmapped:
        strand      = "-" if seq_read.is_reverse else "+"
        strand_code = -1  if seq_read.is_reverse else 1
        ref_name    = seq_read.reference_name or "."
        ref_start   = seq_read.reference_start
        ref_end     = seq_read.reference_end
        qa_start    = seq_read.query_alignment_start
        qa_end      = seq_read.query_alignment_end
        aln_len     = qa_end - qa_start
        if aln_len / seq_read.query_length < coverage_ratio:
            return features_list
        if seq_read.has_tag("NM"):
            if 1.0 - seq_read.get_tag("NM") / aln_len < identity:
                return features_list
        if seq_read.is_reverse:
            seq_start = len(seq) - qa_end
            seq_end   = len(seq) - qa_start
        else:
            seq_start, seq_end = qa_start, qa_end
        q_to_r_poss = get_q2tloc_from_cigar(
            seq_read.cigartuples, strand_code, seq_end - seq_start
        )

    for loc in tsite_locs:
        if not (num_bases <= loc < len(seq) - num_bases):
            continue

        ref_pos = -1
        if not seq_read.is_unmapped:
            if not (seq_start <= loc < seq_end):
                continue
            rpos = q_to_r_poss[loc - seq_start]
            if rpos == -1:
                continue
            ref_pos = (ref_end - 1 - rpos) if strand == "-" else (ref_start + rpos)

        if positions is not None:
            if key_sep.join([ref_name, str(ref_pos), strand]) not in positions:
                continue

        tag = compute_proximity_tag(loc, tag_locs, window=10)

        k_mer     = seq[loc - num_bases: loc + num_bases + 1]
        k_sigs_v  = signal_group[loc - num_bases: loc + num_bases + 1]
        signal_means = np.array([np.mean(x) for x in k_sigs_v], dtype=np.float32)
        signal_stds  = np.array([np.std(x)  for x in k_sigs_v], dtype=np.float32)
        signal_lens  = [len(x) for x in k_sigs_v]

        k_signals_rect = signal_rect_full[loc - num_bases: loc + num_bases + 1]
        # Replace NaN padding with 0.0 for TSV compatibility
        k_signals_rect = np.where(np.isnan(k_signals_rect), 0.0, k_signals_rect)

        features_list.append(
            _features_to_str((
                ref_name,
                str(ref_pos),
                strand,
                ".",
                seq_read.query_name,
                ".",
                k_mer,
                signal_means,
                signal_stds,
                signal_lens,
                k_signals_rect,
                methy_label,
                tag,
            ))
        )

    return features_list


# ─────────────────────────────────────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────────────────────────────────────

def process_sig_seq(
    seq_index,
    files_q,
    feature_Q,
    file_type,
    motif_seqs,
    positions,
    kmer_len,
    signals_len,
    mapq,
    coverage_ratio,
    identity,
    methyloc=0,
    methy_label=1,
    norm_method="mad",
    nproc_extract=1,
    is_single=False,
    plant=False,
):
    LOGGER.info("extract_features process-{} starts".format(os.getpid()))
    while True:
        item = files_q.get()
        if item == "kill":
            files_q.put("kill")
            break

        file_path = item[0]

        def _handle_read(signal, read_name):
            try:
                for seq_read in seq_index.get_alignments(read_name):
                    if seq_read.get_forward_sequence() is None:
                        continue
                    feats = process_data(
                        signal, seq_read, motif_seqs, positions,
                        kmer_len, signals_len, mapq, coverage_ratio, identity,
                        methyloc, methy_label, norm_method, plant,
                    )
                    if feats:
                        while feature_Q.qsize() > (nproc_extract if nproc_extract > 1 else 2) * 3:
                            time.sleep(time_wait)
                        feature_Q.put(feats)
            except KeyError:
                LOGGER.warning("Read %s not found in BAM file", read_name)

        try:
            if file_type == "pod5":
                with pod5.Reader(file_path) as reader:
                    for rec in reader.reads():
                        _handle_read(rec.signal, str(rec.read_id))

            elif file_type in ("slow5", "blow5"):
                s5 = pyslow5.Open(file_path, "r")
                try:
                    for read in s5.seq_reads():
                        _handle_read(read["signal"], read["read_id"])
                finally:
                    s5.close()

            elif file_type == "fast5":
                from .utils import fast5_reader
                if is_single:
                    f5 = fast5_reader.SingleFast5(file_path, is_single=True)
                    try:
                        sig = f5.rescale_signals(f5.get_raw_signal())
                        _handle_read(sig, f5.get_readid())
                    finally:
                        f5.close()
                else:
                    mf = fast5_reader.MultiFast5(file_path)
                    try:
                        for rname in mf:
                            f5 = fast5_reader.SingleFast5(mf[rname], readname=rname)
                            sig = f5.rescale_signals(f5.get_raw_signal())
                            _handle_read(sig, f5.get_readid())
                    finally:
                        mf.close()


        except Exception as e:
            LOGGER.error("Error processing %s: %s", file_path, e)

    LOGGER.info("extract_features process-{} finished".format(os.getpid()))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(args):
    start = time.time()
    LOGGER.info("[extract] starts")

    input_path = os.path.abspath(args.input_dir)
    if not os.path.exists(input_path):
        raise ValueError("--input_dir does not exist: {}".format(input_path))
    if not os.path.isdir(input_path):
        raise NotADirectoryError("--input_dir is not a directory: {}".format(input_path))

    positions  = read_position_file(args.positions) if args.positions else None
    is_dna     = not args.rna
    motif_seqs = get_motif_seqs(args.motifs, is_dna)
    bam_index  = bam_reader.ReadIndexedBam(args.bam)
    is_rec     = str2bool(args.recursively)

    # detect file type and collect files
    file_type = getattr(args, "file_type", None) or detect_file_type(input_path)
    if file_type is None:
        raise ValueError("No signal files (pod5/slow5/blow5/fast5) found in --input_dir")
    is_single = getattr(args, "single", False)
    ext_map   = {"pod5": ".pod5", "slow5": ".slow5", "blow5": ".blow5", "fast5": ".fast5"}
    ext       = ext_map.get(file_type, ".pod5")
    all_files = get_files(input_path, is_rec, ext)
    LOGGER.info("[extract] found %d %s files", len(all_files), file_type)

    files_q  = Queue()
    fill_files_queue(files_q, all_files)
    files_q.put("kill")

    feature_Q = Queue()
    nproc     = max(1, args.nproc - 1)

    plant = getattr(args, "plant", False)
    workers = []
    for proc_idx in range(nproc):
        p = mp.Process(
            target=process_sig_seq,
            args=(
                bam_index, files_q, feature_Q, file_type,
                motif_seqs, positions,
                args.seq_len, args.signal_len,
                args.mapq, args.coverage_ratio, args.identity,
                args.mod_loc, args.methy_label, args.normalize_method,
                nproc, is_single, plant,
            ),
            name="extracter_{:03d}".format(proc_idx),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    p_w = mp.Process(
        target=write_featurestr,
        args=(args.write_path, feature_Q, args.w_batch_num, str2bool(args.w_is_dir)),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()

    for p in workers:
        p.join()
    feature_Q.put("kill")
    p_w.join()

    LOGGER.info("[extract] finished, cost {:.1f}s".format(time.time() - start))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract features from pod5/slow5/blow5 for training or testing.\n"
            "It is suggested to run this module one flowcell at a time."
        )
    )

    g_input = parser.add_argument_group("INPUT")
    g_input.add_argument("--input_dir", "-i", required=True,
                         help="directory of signal files (pod5/slow5/blow5)")
    g_input.add_argument("--single", action="store_true", default=False,
                         help="fast5 files are in single-read format (ignored for pod5/slow5)")
    g_input.add_argument("--recursively", "-r", default="yes",
                         help="search input_dir recursively, default yes")
    g_input.add_argument("--rna", action="store_true", default=False,
                         help="input is RNA (signals reversed). Currently informational only.")
    g_input.add_argument("--bam", required=True, help="BAM filepath (indexed)")
    g_input.add_argument("--reference_path", default=None,
                         help="reference FASTA (optional, informational)")

    g_ext = parser.add_argument_group("EXTRACTION")
    g_ext.add_argument("--normalize_method", choices=["mad", "zscore"], default="mad",
                       help="signal normalisation method, default mad")
    g_ext.add_argument("--methy_label", type=int, choices=[0, 1], default=1,
                       help="label for modified bases (0 or 1), default 1")
    g_ext.add_argument("--seq_len", type=int, default=21,
                       help="k-mer length (must be odd), default 21")
    g_ext.add_argument("--signal_len", type=int, default=15,
                       help="signals per base in rect matrix, default 15")
    g_ext.add_argument("--motifs", default="CG",
                       help="motif(s) to extract, comma-separated, default CG")
    g_ext.add_argument("--plant", action="store_true", default=False,
                       help="plant mode: proximity tag counts any C within "
                            "±10 bp (motif-agnostic). Default (human mode): "
                            "only same-motif sites are counted.")
    g_ext.add_argument("--mod_loc", type=int, default=0,
                       help="0-based position of target base in motif, default 0")
    g_ext.add_argument("--positions", default=None,
                       help="tab-separated file of positions to restrict extraction")

    g_map = parser.add_argument_group("MAPPING FILTERS")
    g_map.add_argument("--mapq", type=int, default=1,
                       help="minimum mapping quality, default 1")
    g_map.add_argument("--identity", type=float, default=0.0,
                       help="minimum alignment identity, default 0.0")
    g_map.add_argument("--coverage_ratio", type=float, default=0.50,
                       help="minimum aligned/query length ratio, default 0.50")

    g_out = parser.add_argument_group("OUTPUT")
    g_out.add_argument("--write_path", "-o", required=True,
                       help="output file path (or directory if --w_is_dir yes)")
    g_out.add_argument("--w_is_dir", default="no",
                       help="write features into multiple files in a directory, default no")
    g_out.add_argument("--w_batch_num", type=int, default=200,
                       help="feature batch size per output file when --w_is_dir yes, default 200")

    parser.add_argument("--nproc", "-p", type=int, default=10,
                        help="number of processes, default 10")

    args = parser.parse_args()
    display_args(args)
    extract_features(args)


if __name__ == "__main__":
    sys.exit(main())
