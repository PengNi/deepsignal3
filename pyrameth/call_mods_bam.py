"""
call_mods_bam.py
Read-level parallel ModBAM inference pipeline (DeepMod2-style).

Architecture (cf. call_modifications.py batch-level pipeline):
  producer_bam()   → [signal_q]  per-read batches
  model_worker_bam() ← [signal_q], → [output_q]  all sites of a read inferred together
  bam_writer()     ← [output_q]  writes MM/ML-tagged BAM

Key difference from call_modifications.py:
  - The unit through the queue is a *read* (not a site).
  - All sites of one read stay together, so MM/ML tags are built
    per read without any cross-read accumulation in the writer.
  - call_modifications.py is untouched.

Selected via CLI:
  pyrameth call_mods       → TSV batch pipeline (unchanged)
  pyrameth call_mods_bam   → this BAM read-level pipeline
"""

from __future__ import absolute_import

import array
import math
import os
import random
import re
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import numpy as np
import pysam
import torch
import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from .models import ModelBiLSTM, modelMTM
from .utils.process_utils import (
    base2code_dna,
    code2base_dna,
    compute_proximity_tag,
    detect_file_type,
    display_args,
    get_files,
    get_logger,
    get_motif_seqs,
    get_refloc_of_methysite_in_motif,
    normalize_signals,
    read_position_file,
    str2bool,
)
from .utils import bam_reader
from .utils_dataloader import (
    _get_ref_coords,
    _group_signals_by_movetable_v2,
    _parse_bam_read,
)
from .call_modifications import load_model_bilstm, load_model_mtm

LOGGER = get_logger(__name__)

_READS_PER_BATCH = 100   # reads per queue item, matching DeepMod2's default
_QUEUE_DEPTH = 200       # max items in each mp.Queue
_MOTIF_BASE = "C"
_MOD_SYMBOL = "m"        # SAM base-modification symbol for 5mC


# ─────────────────────────────────────────────────────────────────────────────
# MM / ML tag construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_mm_ml(site_locs, probs, fwd_seq):
    """
    Build MM tag string and ML byte array.

    site_locs : sorted list of 0-based positions in the *forward* read sequence
    probs     : modification probabilities (prob_1) matching site_locs
    fwd_seq   : forward-strand read sequence (from pysam get_forward_sequence())

    Convention: positions are in the forward-sequence coordinate system, same
    as generate_5mC_modbam_file.py.  The '?' strand flag marks this.
    Returns (mm_str, ml_array) or (None, None) when no sites qualify.
    """
    if not site_locs:
        return None, None

    c_positions = [m.start() for m in re.finditer(_MOTIF_BASE, fwd_seq)]
    if not c_positions:
        return None, None

    c_index = {pos: idx for idx, pos in enumerate(c_positions)}

    base_orders = []
    for loc in site_locs:
        idx = c_index.get(loc)
        if idx is None:
            return None, None   # site not on a C in the forward sequence
        base_orders.append(idx)

    # skip-count encoding: first = absolute index, rest = gap between consecutive entries
    skips = [base_orders[0]]
    for i in range(1, len(base_orders)):
        skips.append(base_orders[i] - base_orders[i - 1] - 1)

    mm_str = f"{_MOTIF_BASE}+{_MOD_SYMBOL}?," + ",".join(str(s) for s in skips) + ";"
    ml_arr = array.array("B", [min(255, math.floor(p * 256)) for p in probs])
    return mm_str, ml_arr


# ─────────────────────────────────────────────────────────────────────────────
# BAM record serialisation / reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_bam_read(read):
    """Extract fields needed to reconstruct an AlignedSegment after IPC."""
    quals = read.query_qualities
    # Strip existing MM/ML and pulse tags; they will be replaced or are unwanted
    tags = [
        (t[0], t[1])
        for t in read.get_tags()
        if t[0] not in {"MM", "ML", "fi", "fp", "ri", "rp"}
    ]
    return {
        "query_name":           read.query_name,
        "flag":                 read.flag,
        "reference_name":       read.reference_name,
        "reference_start":      read.reference_start,
        "mapping_quality":      read.mapping_quality,
        "cigar":                read.cigartuples,
        "next_reference_name":  read.next_reference_name,
        "next_reference_start": read.next_reference_start,
        "template_length":      read.template_length,
        "query_sequence":       read.query_sequence,
        "query_qualities":      quals.tolist() if quals is not None else None,
        "tags":                 tags,
    }


def _write_bam_read(bam_dict, header, mm_str, ml_arr, out_bam):
    """Reconstruct AlignedSegment from dict, attach MM/ML if present, write."""
    seg = pysam.AlignedSegment(header)
    seg.query_name    = bam_dict["query_name"]
    seg.flag          = bam_dict["flag"]

    ref_name = bam_dict["reference_name"]
    if ref_name is not None:
        try:
            seg.reference_id = header.get_tid(ref_name)
        except (ValueError, KeyError):
            seg.reference_id = -1
    else:
        seg.reference_id = -1

    seg.reference_start  = bam_dict["reference_start"] or 0
    seg.mapping_quality  = bam_dict["mapping_quality"] or 0

    if bam_dict["cigar"]:
        seg.cigar = bam_dict["cigar"]

    next_ref = bam_dict["next_reference_name"]
    if next_ref is not None:
        try:
            seg.next_reference_id = header.get_tid(next_ref)
        except (ValueError, KeyError):
            seg.next_reference_id = -1
    else:
        seg.next_reference_id = -1

    seg.next_reference_start = bam_dict["next_reference_start"] or 0
    seg.template_length      = bam_dict["template_length"] or 0
    seg.query_sequence       = bam_dict["query_sequence"]

    if bam_dict["query_qualities"] is not None:
        seg.query_qualities = np.array(bam_dict["query_qualities"], dtype=np.uint8)

    tags = list(bam_dict["tags"])
    if mm_str is not None:
        tags.append(("MM", mm_str))
        tags.append(("ML", ml_arr))
    if tags:
        seg.set_tags(tags)

    out_bam.write(seg)


# ─────────────────────────────────────────────────────────────────────────────
# Per-read feature extraction  (called inside producer)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_read_data(signal, seq_read, motif_seqs, positions, args):
    """
    Extract per-site features for one read and serialise its BAM record.

    Returns a dict:
        bam_dict  : serialised BAM record
        fwd_seq   : forward-strand sequence (for MM tag)
        site_locs : sorted list of 0-based positions in fwd_seq
        features  : list of per-site feature tuples (MTM or BiLSTM)
        has_sites : bool

    Returns None to skip the read entirely (QC fail or bad data).
    """
    is_bilstm = getattr(args, "model_class", "mtm") == "bilstm"

    parsed = _parse_bam_read(signal, seq_read, args)
    if parsed is None:
        return None

    if is_bilstm:
        seq, norm_signal, signal_rect, movetable, stride = parsed
        try:
            signal_group = _group_signals_by_movetable_v2(norm_signal, movetable, stride)
        except (ValueError, AssertionError):
            return None
    else:
        seq, _, signal_rect, _, _ = parsed

    if seq_read.mapping_quality < args.mapq:
        return None

    if not seq_read.is_unmapped:
        qa_start = seq_read.query_alignment_start
        qa_end   = seq_read.query_alignment_end
        if (qa_end - qa_start) / seq_read.query_length < args.coverage_ratio:
            return None

    tsite_locs = get_refloc_of_methysite_in_motif(seq, motif_seqs, args.mod_loc)
    coords     = _get_ref_coords(seq_read, seq)
    strand     = coords["strand"]   if coords else "."
    ref_name   = coords["ref_name"] if coords else "."

    chrom_args = getattr(args, "chrom", None) or []
    _excl = {c[2:] for c in chrom_args if c.startswith("no")}
    _incl = {c      for c in chrom_args if not c.startswith("no")}
    if (_excl and ref_name in _excl) or (_incl and ref_name not in _incl):
        return None

    bam_dict = _serialize_bam_read(seq_read)

    if not tsite_locs:
        return {"bam_dict": bam_dict, "fwd_seq": seq,
                "site_locs": [], "features": [], "has_sites": False}

    num_bases = (args.seq_len - 1) // 2
    plant     = getattr(args, "plant", False)
    tag_locs  = [i for i, b in enumerate(seq) if b == "C"] if plant else tsite_locs

    site_locs = []
    features  = []

    for loc in tsite_locs:
        if not (num_bases <= loc < len(seq) - num_bases):
            continue

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
        else:
            ref_pos = -1

        if positions is not None:
            if f"{ref_name}\t{ref_pos}\t{strand}" not in positions:
                continue

        k_mer     = seq[loc - num_bases: loc + num_bases + 1]
        k_seq     = np.fromiter(
            (base2code_dna[x] for x in k_mer),
            dtype=np.int64, count=args.seq_len,
        )
        k_signals = signal_rect[loc - num_bases: loc + num_bases + 1]

        if is_bilstm:
            k_sigs_v = signal_group[loc - num_bases: loc + num_bases + 1]
            means = np.array([np.mean(x) for x in k_sigs_v], dtype=np.float32)
            stds  = np.array([np.std(x)  for x in k_sigs_v], dtype=np.float32)
            lens  = np.array([len(x)     for x in k_sigs_v], dtype=np.int32)
            features.append((k_seq, k_signals, means, stds, lens))
        else:
            tag = compute_proximity_tag(loc, tag_locs, window=10)
            features.append((k_seq, k_signals, tag))

        site_locs.append(loc)

    return {
        "bam_dict":  bam_dict,
        "fwd_seq":   seq,
        "site_locs": site_locs,
        "features":  features,
        "has_sites": len(site_locs) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Producer
# ─────────────────────────────────────────────────────────────────────────────

def producer_bam(worker_id, files, queues, args, motif_seqs, positions,
                 file_type, num_workers, nproc_io):
    """
    Read signal files, extract per-read features, send batches of
    _READS_PER_BATCH read_data dicts to model workers.
    """
    my_files = files[worker_id::nproc_io]
    print(f"[Producer-{worker_id}] {len(my_files)} {file_type} files", flush=True)

    bam_idx = bam_reader.ReadIndexedBam(args.bam)

    batch = []

    def _flush():
        if batch:
            qid = random.randint(0, num_workers - 1)
            queues[qid].put(list(batch))
            batch.clear()

    def _handle_read(signal, read_name):
        try:
            aligns = list(bam_idx.get_alignments(read_name))
        except KeyError:
            return
        for seq_read in aligns:
            rd = _extract_read_data(signal, seq_read, motif_seqs, positions, args)
            if rd is None:
                continue
            batch.append(rd)
            if len(batch) >= _READS_PER_BATCH:
                _flush()

    for file in my_files:
        try:
            if file_type == "pod5":
                import pod5
                with pod5.Reader(file) as reader:
                    for read in reader.reads():
                        _handle_read(read.signal, str(read.read_id))

            elif file_type in ("slow5", "blow5"):
                import pyslow5
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

    _flush()
    print(f"[Producer-{worker_id}] done", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_batch_mtm(batch_reads, args, device, model):
    """
    Run MTM inference over all sites from a list of read_data dicts.
    Returns list of (bam_dict, mm_str, ml_arr) per read.
    """
    all_k_seq, all_k_sig, all_tags = [], [], []
    boundaries = [0]

    for rd in batch_reads:
        for (k_seq, k_signals, tag) in rd["features"]:
            all_k_seq.append(np.asarray(k_seq,     dtype=np.int64))
            all_k_sig.append(np.asarray(k_signals, dtype=np.float32))
            all_tags.append(tag)
        boundaries.append(boundaries[-1] + len(rd["features"]))

    results = []
    if not all_k_seq:
        for rd in batch_reads:
            results.append((rd["bam_dict"], None, None))
        return results

    k_np = np.stack(all_k_seq)
    s_np = np.stack(all_k_sig)
    B, L, S = s_np.shape

    kmers   = torch.from_numpy(k_np).to(device, non_blocking=True)
    signals = torch.from_numpy(s_np.reshape(B, L * S, 1)).to(device, non_blocking=True)
    tags    = torch.tensor(all_tags, dtype=torch.long, device=device)

    kmer_expand = kmers.unsqueeze(2).expand(-1, -1, S).reshape(B, -1)
    x_mask = torch.isnan(signals)
    false_mask = torch.zeros(
        (*x_mask.shape[:-1], args.n_embed), device=device, dtype=torch.bool
    )
    x_mask   = torch.cat([x_mask, false_mask], dim=-1)
    tpos     = torch.arange(L * S, device=device).unsqueeze(0).expand(B, -1)
    x_static = tags.unsqueeze(-1)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
    ):
        logits = model(signals, kmer_expand, x_mask, tpos, x_static)
        probs  = torch.softmax(logits.float(), dim=-1)

    probs_np = probs.cpu().numpy()  # (total_sites, 2)

    if device.type == "cuda":
        del kmers, signals, tags, logits, probs, kmer_expand, x_mask, false_mask
        torch.cuda.empty_cache()

    for i, rd in enumerate(batch_reads):
        s, e = boundaries[i], boundaries[i + 1]
        if s == e:
            results.append((rd["bam_dict"], None, None))
            continue
        site_probs = probs_np[s:e, 1].tolist()
        mm_str, ml_arr = _build_mm_ml(rd["site_locs"], site_probs, rd["fwd_seq"])
        results.append((rd["bam_dict"], mm_str, ml_arr))

    return results


def _infer_batch_bilstm(batch_reads, device, model):
    """
    Run BiLSTM inference over all sites from a list of read_data dicts.
    Returns list of (bam_dict, mm_str, ml_arr) per read.
    """
    all_k_seq, all_means, all_stds, all_lens, all_k_sig = [], [], [], [], []
    boundaries = [0]

    for rd in batch_reads:
        for (k_seq, k_signals, means, stds, lens) in rd["features"]:
            all_k_seq.append(torch.from_numpy(np.asarray(k_seq,     dtype=np.int64)))
            all_k_sig.append(torch.from_numpy(np.asarray(k_signals, dtype=np.float32)))
            all_means.append(torch.from_numpy(np.asarray(means,     dtype=np.float32)))
            all_stds.append( torch.from_numpy(np.asarray(stds,      dtype=np.float32)))
            all_lens.append( torch.from_numpy(np.asarray(lens,      dtype=np.float32)))
        boundaries.append(boundaries[-1] + len(rd["features"]))

    results = []
    if not all_k_seq:
        for rd in batch_reads:
            results.append((rd["bam_dict"], None, None))
        return results

    kmers   = torch.stack(all_k_seq).to(device, non_blocking=True)
    means   = torch.stack(all_means).to(device, non_blocking=True)
    stds    = torch.stack(all_stds).to(device,  non_blocking=True)
    lens    = torch.stack(all_lens).to(device,  non_blocking=True)
    signals = torch.stack(all_k_sig).to(device, non_blocking=True)

    with torch.inference_mode():
        _, probs = model(kmers.long(), means, stds, lens, signals)

    probs_np = probs.cpu().numpy()

    if device.type == "cuda":
        del kmers, means, stds, lens, signals, probs
        torch.cuda.empty_cache()

    for i, rd in enumerate(batch_reads):
        s, e = boundaries[i], boundaries[i + 1]
        if s == e:
            results.append((rd["bam_dict"], None, None))
            continue
        site_probs = probs_np[s:e, 1].tolist()
        mm_str, ml_arr = _build_mm_ml(rd["site_locs"], site_probs, rd["fwd_seq"])
        results.append((rd["bam_dict"], mm_str, ml_arr))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model worker
# ─────────────────────────────────────────────────────────────────────────────

def model_worker_bam(rank, device, queue, output_q, args, nproc_io):
    """
    Consume read batches from queue, run inference, push per-read
    (bam_dict, mm_str, ml_arr) lists to output_q.
    """
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_num_threads(max(1, args.cpu_threads_per_worker))

    is_mtm = (args.model_class == "mtm")
    model  = load_model_mtm(args, device) if is_mtm else load_model_bilstm(args, device)
    model.eval()

    end_count = 0
    while True:
        item = queue.get()
        if item is None:
            end_count += 1
            if end_count == nproc_io:
                break
            continue

        # item is a list of read_data dicts (_READS_PER_BATCH reads)
        try:
            if is_mtm:
                results = _infer_batch_mtm(item, args, device, model)
            else:
                results = _infer_batch_bilstm(item, device, model)
            output_q.put(results)
        except Exception as e:
            print(f"[Worker-{rank}] inference error: {e}", flush=True)

    print(f"[Worker-{rank}({device})] done", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# BAM writer
# ─────────────────────────────────────────────────────────────────────────────

def bam_writer(output_bam_path, input_bam_path, output_q, sort_index):
    """
    Write AlignedSegments with MM/ML tags to output BAM.
    Receives lists of (bam_dict, mm_str, ml_arr) from output_q.
    Optionally sorts and indexes the result.
    """
    pysam.set_verbosity(0)
    with pysam.AlignmentFile(input_bam_path, "rb", check_sq=False) as tmpl:
        header = tmpl.header.to_dict()

    cnt_w = cnt_mm = 0
    with pysam.AlignmentFile(
        output_bam_path, "wb", header=pysam.AlignmentHeader.from_dict(header)
    ) as out_bam:
        hdr = out_bam.header
        while True:
            item = output_q.get()
            if item == "kill":
                break
            for bam_dict, mm_str, ml_arr in item:
                _write_bam_read(bam_dict, hdr, mm_str, ml_arr, out_bam)
                cnt_w += 1
                if mm_str is not None:
                    cnt_mm += 1

    print(
        f"[BAM-Writer] wrote {cnt_w} reads, {cnt_mm} with MM/ML tags",
        flush=True,
    )

    if sort_index and output_bam_path.endswith(".bam"):
        print("[BAM-Writer] sorting and indexing...", flush=True)
        sorted_tmp = output_bam_path + ".sorted_tmp.bam"
        pysam.sort("-o", sorted_tmp, output_bam_path)
        os.replace(sorted_tmp, output_bam_path)
        pysam.index(output_bam_path)
        print("[BAM-Writer] done.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def inference_bam(args):
    """
    Read-level parallel BAM inference pipeline.
    Signal input only (pod5 / slow5 / fast5); no TSV input path.
    """
    mp.set_start_method("spawn", force=True)

    # ── device selection ──────────────────────────────────────────────────────
    num_gpu  = torch.cuda.device_count()
    use_gpu  = num_gpu > 0 and not getattr(args, "use_cpu", False)

    if use_gpu:
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpu)]
        LOGGER.info(f"Using {num_gpu} GPU(s): {devices}")
    else:
        nproc_cpu = max(1, getattr(args, "nproc_cpu", 1))
        devices   = [torch.device("cpu")] * nproc_cpu
        n_cores   = os.cpu_count() or 1
        args.cpu_threads_per_worker = max(1, n_cores // nproc_cpu)
        LOGGER.info(
            f"CPU mode: {nproc_cpu} worker(s), "
            f"{args.cpu_threads_per_worker} thread(s) each."
        )

    n_workers = len(devices)
    nproc_io  = max(1, args.nproc)

    # ── detect signal files ───────────────────────────────────────────────────
    file_type  = detect_file_type(args.input_path, True)
    files      = get_files(args.input_path, True, file_type)
    motif_seqs = get_motif_seqs(args.motifs, True)
    positions  = read_position_file(args.positions) if args.positions else None
    LOGGER.info(f"Signal input: {len(files)} {file_type} files, {nproc_io} IO worker(s)")

    # ── queues ────────────────────────────────────────────────────────────────
    queues   = [mp.Queue(_QUEUE_DEPTH) for _ in range(n_workers)]
    output_q = mp.Queue(_QUEUE_DEPTH)

    # ── producers ─────────────────────────────────────────────────────────────
    producers = []
    for i in range(nproc_io):
        p = mp.Process(
            target=producer_bam,
            args=(i, files, queues, args, motif_seqs, positions,
                  file_type, n_workers, nproc_io),
        )
        p.start()
        producers.append(p)

    # ── model workers ─────────────────────────────────────────────────────────
    workers = []
    for rank, device in enumerate(devices):
        p = mp.Process(
            target=model_worker_bam,
            args=(rank, device, queues[rank], output_q, args, nproc_io),
        )
        p.start()
        workers.append(p)

    # ── BAM writer ────────────────────────────────────────────────────────────
    sort_index = getattr(args, "sort_bam", True)
    p_writer   = mp.Process(
        target=bam_writer,
        args=(args.output_bam, args.bam, output_q, sort_index),
    )
    p_writer.start()

    # ── join ──────────────────────────────────────────────────────────────────
    for p in producers:
        p.join()
    LOGGER.info("All producers done.")

    for q in queues:
        for _ in range(nproc_io):
            q.put(None)

    for p in workers:
        p.join()

    output_q.put("kill")
    p_writer.join()

    LOGGER.info("ALL DONE")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI (also reachable via pyrameth call_mods_bam)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "pyrameth call_mods_bam – read-level BAM output pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p_in = parser.add_argument_group("INPUT")
    p_in.add_argument("--input_path", "-i", required=True,
                      help="Signal directory (pod5/slow5/fast5)")
    p_in.add_argument("--bam", required=True,
                      help="Input BAM file (aligned)")
    p_in.add_argument("--recursively", "-r", type=str, default="yes")

    p_out = parser.add_argument_group("OUTPUT")
    p_out.add_argument("--output_bam", "-o", required=True,
                       help="Output ModBAM path (.bam)")
    p_out.add_argument("--sort_bam", type=str2bool, default="yes",
                       help="Sort and index output BAM when done, default yes")

    p_model = parser.add_argument_group("MODEL")
    p_model.add_argument("--model_path", "-m", required=True)
    p_model.add_argument("--model_class", type=str, default="mtm",
                         choices=["mtm", "bilstm"])
    p_model.add_argument("--use_compile", type=str2bool, default="no")
    p_model.add_argument("--use_cpu", action="store_true", default=False)
    p_model.add_argument("--nproc_cpu", type=int, default=1)

    p_hp = parser.add_argument_group("MODEL_HYPER")
    p_hp.add_argument("--seq_len",      type=int,   default=21)
    p_hp.add_argument("--signal_len",   type=int,   default=15)
    p_hp.add_argument("--class_num",    type=int,   default=2)
    p_hp.add_argument("--dropout_rate", type=float, default=0.0)
    p_hp.add_argument("--n_vocab",      type=int,   default=16)
    p_hp.add_argument("--n_embed",      type=int,   default=4)

    p_lstm = parser.add_argument_group("BILSTM_HYPER")
    p_lstm.add_argument("--hid_rnn",      type=int, default=256)
    p_lstm.add_argument("--layernum1",    type=int, default=3)
    p_lstm.add_argument("--layernum2",    type=int, default=1)
    p_lstm.add_argument("--is_base",      type=str, default="yes")
    p_lstm.add_argument("--is_signallen", type=str, default="yes")
    p_lstm.add_argument("--is_trace",     type=str, default="no")

    p_mtm = parser.add_argument_group("MTM_HYPER")
    p_mtm.add_argument("--mtm_num_base_features", type=int,           default=1)
    p_mtm.add_argument("--mtm_hid_rnn",           type=int,           default=128)
    p_mtm.add_argument("--mtm_d_static",          type=int,           default=1)
    p_mtm.add_argument("--mtm_ratios",   nargs="+", type=int,         default=[2, 2, 2, 2])
    p_mtm.add_argument("--mtm_r_hid",             type=int,           default=4)
    p_mtm.add_argument("--mtm_norm_first",  type=str2bool,            default="True")
    p_mtm.add_argument("--mtm_down_mode",   type=str,                 default="concat",
                       choices=["concat", "avg", "max"])
    p_mtm.add_argument("--mtm_temporal_depth", type=int,              default=2)

    p_ext = parser.add_argument_group("EXTRACTION")
    p_ext.add_argument("--motifs",           type=str,   default="CG")
    p_ext.add_argument("--mod_loc",          type=int,   default=0)
    p_ext.add_argument("--methy_label",      type=int,   default=1, choices=[0, 1])
    p_ext.add_argument("--normalize_method", type=str,   default="mad",
                       choices=["mad", "zscore"])
    p_ext.add_argument("--mapq",             type=int,   default=1)
    p_ext.add_argument("--coverage_ratio",   type=float, default=0.5)
    p_ext.add_argument("--identity",         type=float, default=0.0)
    p_ext.add_argument("--positions",        type=str,   default=None)
    p_ext.add_argument("--rna",              action="store_true", default=False)
    p_ext.add_argument("--plant",            action="store_true", default=False,
                       help="Plant mode: proximity tag counts any C within ±10 bp.")
    p_ext.add_argument("--chrom",            type=str, nargs="+", default=None)

    p_perf = parser.add_argument_group("PERFORMANCE")
    p_perf.add_argument("--batch_size", "-b", type=int, default=500)
    p_perf.add_argument("--nproc",      "-p", type=int, default=10,
                        help="Number of IO producer processes")

    args = parser.parse_args()
    display_args(args)
    inference_bam(args)


if __name__ == "__main__":
    sys.exit(main())
