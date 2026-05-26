#! /usr/bin/env python
"""
calculate modification frequency at genome level.

Two modes:
  - count mode  (default): pure count-based aggregation, TSV or bedMethyl output.
  - aggregate mode (--aggre_model): neural-network refinement via AggrAttRNN,
    always writes bedMethyl.
"""

from __future__ import absolute_import

import argparse
import os
import sys
import time
import gzip

from .utils.txt_formater import ModRecord, SiteStats, split_key


# ─────────────────────────────────────────────
# Shared file-collection helper
# ─────────────────────────────────────────────

def _collect_files(input_paths, file_uid=None):
    mods_files = []
    for ipath in input_paths:
        input_path = os.path.abspath(ipath)
        if os.path.isdir(input_path):
            for ifile in os.listdir(input_path):
                if file_uid is None or ifile.find(file_uid) != -1:
                    mods_files.append(os.path.join(input_path, ifile))
        elif os.path.isfile(input_path):
            mods_files.append(input_path)
        else:
            raise ValueError("input_path not found: {}".format(input_path))
    return mods_files


def _open_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, 'rt')
    return open(path, 'r')


# ─────────────────────────────────────────────
# Mode 1: count-based frequency
# ─────────────────────────────────────────────

def calculate_mods_frequency(mods_files, prob_cf):
    sitekeys = set()
    sitekey2stats = dict()
    count, used = 0, 0

    for mods_file in mods_files:
        with _open_file(mods_file) as infile:
            for line in infile:
                words = line.strip().split("\t")
                mod_record = ModRecord(words)
                if mod_record.is_record_callable(prob_cf):
                    key = mod_record._site_key
                    if key not in sitekeys:
                        sitekeys.add(key)
                        sitekey2stats[key] = SiteStats(
                            mod_record._strand,
                            mod_record._pos_in_strand,
                            mod_record._kmer,
                        )
                    sitekey2stats[key]._prob_0 += mod_record._prob_0
                    sitekey2stats[key]._prob_1 += mod_record._prob_1
                    sitekey2stats[key]._coverage += 1
                    if mod_record._called_label == 1:
                        sitekey2stats[key]._met += 1
                    else:
                        sitekey2stats[key]._unmet += 1
                    used += 1
                count += 1

    print("{:.2f}% ({} of {}) calls used..".format(
        used / float(count) * 100, used, count))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file, is_sort, is_bed):
    if is_sort:
        keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))
    else:
        keys = list(sitekey2stats.keys())

    with open(result_file, "w") as wf:
        for key in keys:
            chrom, pos = split_key(key)
            s = sitekey2stats[key]
            assert s._coverage == (s._met + s._unmet)
            if s._coverage == 0:
                print("{} {} has no coverage..".format(chrom, pos))
                continue
            rmet = float(s._met) / s._coverage
            if is_bed:
                wf.write("\t".join([
                    chrom, str(pos), str(pos + 1), ".",
                    str(s._coverage), s._strand,
                    str(pos), str(pos + 1), "0,0,0",
                    str(s._coverage),
                    str(int(round(rmet * 100 + 0.001, 0))),
                ]) + "\n")
            else:
                wf.write(
                    "%s\t%d\t%s\t%s\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n"
                    % (chrom, pos, s._strand, s._pos_in_strand,
                       s._prob_0, s._prob_1,
                       s._met, s._unmet, s._coverage, rmet, s._kmer)
                )


# ─────────────────────────────────────────────
# Mode 2: AggrAttRNN-based refinement
# ─────────────────────────────────────────────

def _get_normalized_histo(probs, binsize=20):
    import numpy as np
    if len(probs) == 0:
        return None
    hist, _ = np.histogram(probs, bins=binsize, range=[0., 1.])
    norm = np.linalg.norm(hist)
    return np.round(hist / (norm + 1e-8), 6)


def _prepare_data(mods_files, prob_cf=0.0, cov_cf=4, bin_size=20):
    """Read per-read calls from one or more TSV files, group by (chrom, pos).

    Returns all positions for inference plus a separate anchor set (cov >= cov_cf)
    used as window context so low-coverage sites never pollute their neighbors.
    """
    from collections import defaultdict, Counter
    import numpy as np

    chrom_pos_data = defaultdict(lambda: defaultdict(lambda: {'probs': [], 'strands': []}))
    count, used = 0, 0

    for mods_file in mods_files:
        with _open_file(mods_file) as f:
            for line in f:
                words = line.strip().split('\t')
                if len(words) < 9:
                    continue
                try:
                    mod_record = ModRecord(words)
                except (ValueError, IndexError):
                    continue
                if not mod_record.is_record_callable(prob_cf):
                    count += 1
                    continue
                chrom_pos_data[mod_record._chromosome][mod_record._pos][
                    'probs'].append(mod_record._prob_1)
                chrom_pos_data[mod_record._chromosome][mod_record._pos][
                    'strands'].append(mod_record._strand)
                count += 1
                used += 1

    print("{:.2f}% ({} of {}) calls used..".format(
        used / float(count) * 100 if count else 0, used, count))

    result = {}
    for chrom, pos_dict in chrom_pos_data.items():
        positions, histograms, coverages, strands = [], [], [], []
        anchor_positions, anchor_histograms = [], []

        for refpos in sorted(pos_dict.keys()):
            item = pos_dict[refpos]
            cov = len(item['probs'])
            hist = _get_normalized_histo(item['probs'], bin_size)
            if hist is None:
                continue
            strand = Counter(item['strands']).most_common(1)[0][0]

            positions.append(refpos)
            histograms.append(hist)
            coverages.append(cov)
            strands.append(strand)

            if cov >= cov_cf:
                anchor_positions.append(refpos)
                anchor_histograms.append(hist)

        if positions:
            result[chrom] = {
                'positions': positions,
                'histograms': histograms,
                'coverages': coverages,
                'strands': strands,
                'anchor_positions': anchor_positions,
                'anchor_histograms': anchor_histograms,
            }
    return result


def _run_aggr_model(positions, histograms, anchor_positions, anchor_histograms,
                    model, seq_len=11, batch_size=1024):
    """Run AggrAttRNN on all positions using only anchor positions as window context.

    Anchor positions (cov >= cov_cf) supply the neighborhood histograms so that
    low-coverage sites never pollute the context of their neighbors, while still
    receiving a model-refined prediction themselves.
    """
    import numpy as np
    import torch

    if not positions:
        return []

    pad_len = seq_len // 2
    bin_size = histograms[0].shape[0]
    device = next(model.parameters()).device

    all_pos_arr = np.array(positions, dtype=np.int64)
    all_hist_arr = np.stack(histograms).astype(np.float32)
    N = len(positions)

    M = len(anchor_positions)
    if M > 0:
        anchor_arr = np.array(anchor_positions, dtype=np.int64)
        anchor_hist_arr = np.stack(anchor_histograms).astype(np.float32)
    else:
        anchor_arr = np.array([], dtype=np.int64)
        anchor_hist_arr = np.zeros((0, bin_size), dtype=np.float32)

    # Pad anchor arrays so boundary positions always have pad_len neighbors
    pad_pos_l = all_pos_arr[0] - 10000
    pad_pos_r = all_pos_arr[-1] + 10000
    padded_anchor_pos = np.concatenate([
        np.full(pad_len, pad_pos_l, dtype=np.int64),
        anchor_arr,
        np.full(pad_len, pad_pos_r, dtype=np.int64),
    ])
    padded_anchor_hist = np.concatenate([
        np.zeros((pad_len, bin_size), dtype=np.float32),
        anchor_hist_arr,
        np.zeros((pad_len, bin_size), dtype=np.float32),
    ], axis=0)

    # For each position find its insertion index in anchor_arr (sorted)
    insert_idx = np.searchsorted(anchor_arr, all_pos_arr)  # (N,)

    # Detect which positions are themselves anchors so we skip self as neighbor
    if M > 0:
        safe_idx = np.minimum(insert_idx, M - 1)
        is_anchor = (insert_idx < M) & (anchor_arr[safe_idx] == all_pos_arr)
    else:
        is_anchor = np.zeros(N, dtype=bool)

    # Left neighbors in padded array: indices [k, k+1, ..., k+pad_len-1]
    # Right neighbors: [k+pad_len, ...] for non-anchors, [k+pad_len+1, ...] for anchors
    right_start = insert_idx + np.where(is_anchor, pad_len + 1, pad_len)  # (N,)

    left_idx  = insert_idx[:, None]  + np.arange(pad_len, dtype=np.int64)[None, :]  # (N, pad_len)
    right_idx = right_start[:, None] + np.arange(pad_len, dtype=np.int64)[None, :]  # (N, pad_len)

    left_hists  = padded_anchor_hist[left_idx]   # (N, pad_len, bin_size)
    right_hists = padded_anchor_hist[right_idx]  # (N, pad_len, bin_size)

    hist_windows = np.concatenate(
        [left_hists, all_hist_arr[:, None, :], right_hists], axis=1
    )  # (N, seq_len, bin_size)

    left_pos  = padded_anchor_pos[left_idx]   # (N, pad_len)
    right_pos = padded_anchor_pos[right_idx]  # (N, pad_len)
    window_pos = np.concatenate(
        [left_pos, all_pos_arr[:, None], right_pos], axis=1
    )  # (N, seq_len)
    pos_dist = np.abs(window_pos - all_pos_arr[:, None]).astype(np.float32)

    new_probs = []
    for i in range(0, N, batch_size):
        b_hist = torch.from_numpy(hist_windows[i:i + batch_size]).to(device)
        b_pos  = torch.from_numpy(pos_dist[i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(b_pos, b_hist)
            probs = outputs.clamp(0.0, 1.0).cpu().numpy().flatten()
            new_probs.extend(np.round(probs, 6).tolist())
    return new_probs


def _write_bedmethyl_aggr(data_dict, refined_probs_dict, output_file, is_sort=False):
    chroms = sorted(data_dict.keys()) if is_sort else list(data_dict.keys())
    with open(output_file, 'w') as f:
        for chrom in chroms:
            info = data_dict[chrom]
            probs = refined_probs_dict.get(chrom, [])
            for i, pos in enumerate(info['positions']):
                if pos < 0:
                    continue
                cov = info['coverages'][i]
                strand = info['strands'][i]
                prob = probs[i]
                perc = int(round(prob * 100 + 0.001, 0))
                f.write("\t".join([
                    chrom, str(pos), str(pos + 1), ".",
                    str(cov), strand,
                    str(pos), str(pos + 1), "0,0,0",
                    str(cov), str(perc),
                ]) + "\n")
    print("bedMethyl written: {}".format(output_file))


# ─────────────────────────────────────────────
# Mode 3: ReadCalibRNN-based read-level calibration
# ─────────────────────────────────────────────

def _build_read_windows(records, K=11):
    """Group TSV records by readname, build K-window features per (read, site).

    records: list of (chrom, pos, prob_1, line_idx) already sorted within each read by pos.
    Returns arrays: window_probs, window_offsets, window_valid, read_feats, line_indices
    each aligned so that result[i] corresponds to records[line_idx[i]].
    """
    import numpy as np
    from collections import defaultdict

    # group by (readname) — caller guarantees records are pre-grouped by readname
    # here we receive one read's records at a time via the generator
    raise NotImplementedError("Use _iter_read_windows instead")


def _run_read_calib_model(mods_files, model, K=11, batch_size=4096):
    """Two-pass streaming inference for ReadCalibRNN.

    Pass 1: stream TSV → build per-readname dict {readname: [(chrom, pos, prob_1, orig_line)]}.
            Done per-chromosome to bound memory.
    Pass 2: for each read build K-window, run model in batches, collect calibrated probs.

    Returns:
        orig_lines: list[str]       — original TSV lines (in file order)
        calib_prob1: np.ndarray     — calibrated prob_1 for each line (same order)
    """
    import numpy as np
    from collections import defaultdict

    pad_len = K // 2

    # ── Pass 1: collect all records ──────────────────────────────────────────
    # {readname: [(chrom, pos, prob_1, global_line_idx)]}
    read_dict = defaultdict(list)
    orig_lines = []

    for mods_file in mods_files:
        with _open_file(mods_file) as f:
            for line in f:
                line = line.rstrip('\n')
                words = line.split('\t')
                if len(words) < 10:
                    orig_lines.append(line)
                    continue
                try:
                    chrom    = words[0]
                    pos      = int(words[1])
                    readname = words[4]
                    prob_1   = float(words[7])
                except (ValueError, IndexError):
                    orig_lines.append(line)
                    continue
                idx = len(orig_lines)
                read_dict[readname].append((chrom, pos, prob_1, idx))
                orig_lines.append(line)

    # sort each read's sites by (chrom, pos)
    for rn in read_dict:
        read_dict[rn].sort(key=lambda t: (t[0], t[1]))

    # ── Pass 2: build windows + run inference ────────────────────────────────
    calib_prob1 = np.full(len(orig_lines), -1.0, dtype=np.float32)
    device = next(model.parameters()).device

    b_wprobs, b_woffs, b_wvalid, b_rfeats, b_lidx = [], [], [], [], []

    def _flush_batch():
        if not b_wprobs:
            return
        t_wprobs  = torch.tensor(np.stack(b_wprobs),  dtype=torch.float32, device=device)
        t_woffs   = torch.tensor(np.stack(b_woffs),   dtype=torch.float32, device=device)
        t_wvalid  = torch.tensor(np.stack(b_wvalid),  dtype=torch.float32, device=device)
        t_rfeats  = torch.tensor(np.stack(b_rfeats),  dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(t_wprobs, t_woffs, t_wvalid, t_rfeats).squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
        for p, li in zip(probs, b_lidx):
            calib_prob1[li] = float(np.round(p, 6))
        b_wprobs.clear(); b_woffs.clear(); b_wvalid.clear()
        b_rfeats.clear(); b_lidx.clear()

    import math
    for readname, sites in read_dict.items():
        n_cpg    = len(sites)
        probs_r  = [s[2] for s in sites]
        mean_p   = float(np.mean(probs_r))
        std_p    = float(np.std(probs_r))
        rf       = [math.log1p(n_cpg), mean_p, std_p]

        chroms_r = [s[0] for s in sites]
        pos_r    = [s[1] for s in sites]

        for ci in range(n_cpg):
            tgt_chrom = chroms_r[ci]
            tgt_pos   = pos_r[ci]
            line_idx  = sites[ci][3]

            wp, wo, wv = [], [], []
            for k in range(-pad_len, pad_len + 1):
                j = ci + k
                if 0 <= j < n_cpg and chroms_r[j] == tgt_chrom:
                    wp.append(probs_r[j])
                    wo.append(float(abs(pos_r[j] - tgt_pos)))
                    wv.append(1.0)
                else:
                    wp.append(0.0)
                    wo.append(0.0)
                    wv.append(0.0)

            b_wprobs.append(wp)
            b_woffs.append(wo)
            b_wvalid.append(wv)
            b_rfeats.append(rf)
            b_lidx.append(line_idx)

            if len(b_wprobs) >= batch_size:
                _flush_batch()

    _flush_batch()
    return orig_lines, calib_prob1


def _write_calibrated_tsv(orig_lines, calib_prob1, output_file):
    """Write calibrated TSV: same as input but with updated prob_0/prob_1/called_label."""
    with open(output_file, 'w') as wf:
        for i, line in enumerate(orig_lines):
            p1 = calib_prob1[i]
            if p1 < 0:          # line was not calibrated (parse error / short)
                wf.write(line + '\n')
                continue
            words = line.split('\t')
            p0 = round(1.0 - p1, 6)
            words[6] = str(p0)
            words[7] = str(round(p1, 6))
            words[8] = '1' if p1 >= 0.5 else '0'
            wf.write('\t'.join(words) + '\n')
    print("calibrated TSV written: {}".format(output_file))


def call_read_calib_to_file(args):
    """Entry point for read-level calibration (Phase 1.5)."""
    import torch
    from collections import OrderedDict
    from .models import ReadCalibRNN

    print("[call_read_calib] start..")
    start = time.time()

    mods_files = _collect_files(args.input_path, getattr(args, 'file_uid', None))
    print("get {} input file(s)..".format(len(mods_files)))

    K           = getattr(args, 'read_calib_seq_len', 11)
    hidden      = getattr(args, 'read_calib_hidden', 32)
    model_type  = getattr(args, 'read_calib_model_type', 'attbigru')
    batch_size  = getattr(args, 'batch_size', 4096)

    print("loading ReadCalibRNN from {}..".format(args.read_calib_model))
    model = ReadCalibRNN(seq_len=K, num_layers=1, num_classes=1,
                         dropout_rate=0, hidden_size=hidden,
                         n_read_feats=3, model_type=model_type)
    checkpoint = torch.load(args.read_calib_model, map_location='cpu', weights_only=True)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        new_sd = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in checkpoint.items()
        )
        model.load_state_dict(new_sd)
    model.eval()

    print("running ReadCalibRNN inference..")
    orig_lines, calib_prob1 = _run_read_calib_model(
        mods_files, model, K=K, batch_size=batch_size)

    print("writing calibrated TSV..")
    _write_calibrated_tsv(orig_lines, calib_prob1, args.result_file)

    print("[call_read_calib] costs {:.1f} seconds..".format(time.time() - start))


# ─────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────

def call_mods_frequency_to_file(args):
    print("[call_freq] start..")
    start = time.time()

    mods_files = _collect_files(args.input_path, getattr(args, 'file_uid', None))
    print("get {} input file(s)..".format(len(mods_files)))

    aggre_model_path = getattr(args, 'aggre_model', None)

    if aggre_model_path:
        # ── aggregate (neural-network refinement) mode ──────────────────────
        import torch
        from collections import OrderedDict
        from .models import AggrAttRNN

        cov_cf   = getattr(args, 'cov_cf', 4)
        bin_size = getattr(args, 'bin_size', 20)
        prob_cf  = getattr(args, 'prob_cf', 0.0)
        is_sort  = getattr(args, 'sort', False)

        aggre_hidden = getattr(args, 'aggre_hidden', 32)
        print("loading aggregate model from {}..".format(aggre_model_path))
        model = AggrAttRNN(seq_len=11, num_layers=1, num_classes=1,
                           dropout_rate=0, hidden_size=aggre_hidden,
                           binsize=bin_size, model_type='attbigru', device='cpu')
        checkpoint = torch.load(aggre_model_path, map_location='cpu', weights_only=True)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            new_sd = OrderedDict(
                (k[7:] if k.startswith('module.') else k, v)
                for k, v in checkpoint.items()
            )
            model.load_state_dict(new_sd)
        model.eval()

        print("reading input files..")
        data_dict = _prepare_data(mods_files, prob_cf=prob_cf,
                                  cov_cf=cov_cf, bin_size=bin_size)

        print("running AggrAttRNN inference..")
        refined = {}
        for chrom, info in data_dict.items():
            refined[chrom] = _run_aggr_model(
                info['positions'], info['histograms'],
                info['anchor_positions'], info['anchor_histograms'],
                model,
            )

        print("writing bedMethyl..")
        _write_bedmethyl_aggr(data_dict, refined, args.result_file, is_sort=is_sort)

    else:
        # ── count-based frequency mode (original call_freq) ──────────────────
        prob_cf = getattr(args, 'prob_cf', 0.0)
        is_sort = getattr(args, 'sort', False)
        is_bed  = getattr(args, 'bed', False)

        print("reading input files..")
        sites_stats = calculate_mods_frequency(mods_files, prob_cf)
        print("writing result..")
        write_sitekey2stats(sites_stats, args.result_file, is_sort, is_bed)

    print("[call_freq] costs %.1f seconds.." % (time.time() - start))


# ─────────────────────────────────────────────
# CLI (standalone usage)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="calculate frequency of interested sites at genome level. "
                    "With --aggre_model, uses AggrAttRNN neural-network refinement."
    )
    parser.add_argument("--input_path", "-i", action="append", type=str, required=True,
                        help="output file(s) from call_mods, or a directory. "
                             "Can be used multiple times.")
    parser.add_argument("--file_uid", type=str, default=None,
                        help="unique string shared by all target files in a directory")
    parser.add_argument("--result_file", "-o", type=str, required=True,
                        help="output file path")
    parser.add_argument("--bed", action="store_true", default=False,
                        help="save in bedMethyl format (count mode only)")
    parser.add_argument("--sort", action="store_true", default=False,
                        help="sort output by chromosome and position")
    parser.add_argument("--prob_cf", type=float, default=0.0,
                        help="remove ambiguous calls where |prob1-prob0| < prob_cf, default 0.0")
    # aggregate-mode options
    parser.add_argument("--aggre_model", "-m", type=str, default=None,
                        help="AggrAttRNN model checkpoint (.ckpt). "
                             "When provided, uses neural-network refinement and always writes bedMethyl.")
    parser.add_argument("--cov_cf", type=int, default=4,
                        help="minimum read coverage per site for aggregate mode, default 4")
    parser.add_argument("--bin_size", type=int, default=20,
                        help="histogram bin count for aggregate mode, default 20")
    parser.add_argument("--aggre_hidden", type=int, default=32,
                        help="hidden size of AggrAttRNN, must match the trained model, default 32")

    args = parser.parse_args()
    call_mods_frequency_to_file(args)


def main_read_calib():
    """Standalone CLI for read-level calibration (Phase 1.5)."""
    parser = argparse.ArgumentParser(
        description="ReadCalibRNN read-level calibration. "
                    "Takes per-read call TSV from call_mods and outputs a "
                    "calibrated TSV with updated prob_0/prob_1/called_label."
    )
    parser.add_argument("--input_path", "-i", action="append", type=str, required=True,
                        help="per-read call TSV file(s) or directory. Can be used multiple times.")
    parser.add_argument("--file_uid", type=str, default=None,
                        help="unique string shared by all target files in a directory")
    parser.add_argument("--result_file", "-o", type=str, required=True,
                        help="output calibrated TSV file path")
    parser.add_argument("--read_calib_model", "-m", type=str, required=True,
                        help="ReadCalibRNN checkpoint (.ckpt)")
    parser.add_argument("--read_calib_hidden", type=int, default=32,
                        help="hidden size, must match trained model, default 32")
    parser.add_argument("--read_calib_seq_len", type=int, default=11,
                        help="window size K, must match trained model, default 11")
    parser.add_argument("--read_calib_model_type", type=str, default="attbigru",
                        choices=["attbigru", "attbilstm"],
                        help="model architecture, default attbigru")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="inference batch size, default 4096")

    args = parser.parse_args()
    call_read_calib_to_file(args)


if __name__ == "__main__":
    sys.exit(main())
