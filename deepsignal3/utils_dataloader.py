import torch
from torch.utils.data import IterableDataset
from torch.multiprocessing import Queue
import h5py  # Assuming fast5 files are in HDF5 format, update accordingly
import pod5
import numpy as np
import time
import os
import random
import math

from .utils.process_utils import get_logger
from .utils.process_utils import fill_files_queue
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import normalize_signals
from .utils.process_utils import base2code_dna,complement_seq
from .utils.process_utils import CIGAR2CODE,CIGAR_REGEX

from .utils.constants_torch import FloatTensor
from .utils import fast5_reader

import mappy
import gzip

from collections import namedtuple

LOGGER = get_logger(__name__)

time_wait = 0.01
key_sep = "||"

MAP_RES = namedtuple(
    "MAP_RES",
    (
        "read_id",
        "q_seq",
        "ref_seq",
        "ctg",
        "strand",
        "r_st",
        "r_en",
        "q_st",
        "q_en",
        "cigar",
        "mapq",
    ),
)

class Pod5Dataset(IterableDataset):
    def __init__(self, pod5_dr, bam_index, motif_seqs, positions,device, args):
        super(Pod5Dataset).__init__()
        self.files = pod5_dr
        self.bam_index = bam_index
        self.motif_seqs = motif_seqs
        self.positions = positions
        self.args = args
        self.device = device

    def __iter__(self):
        # 填充pod5数据队列
        pod5s_q = []
        fill_files_queue(pod5s_q, self.files)
        LOGGER.info("extract_features process-{} starts".format(os.getpid()))
        for files in pod5s_q:
            for file in files:  # Iterate over files
                with pod5.Reader(file) as reader:
                # Using list to collect data instead of yielding a generator
                    for read_record in reader.reads():
                        data_list = self.process_sig_seq(read_record)
                        for item in data_list:
                            if len(item)==0:
                                #LOGGER.info("empty item")
                                continue
                            yield to_tensor(item,self.device)
    def process_sig_seq(self,
        read_record,
    ):       
        #results = []  # Collect results in a list                 
        read_name = str(read_record.read_id)
        signal = read_record.signal
        if signal is None:
            return []
        try:
            for seq_read in self.bam_index.get_alignments(read_name):
                seq = seq_read.get_forward_sequence()
                if seq is None:
                    continue
                data = (signal, seq_read)
                feature_lists = self.process_data(
                    data,
                )
                return feature_lists # Append to list instead of yielding
        except KeyError:
            LOGGER.warning("Read:%s not found in BAM file" % read_name)
            return []
        #LOGGER.info("extract_features process-{} finished".format(os.getpid()))
        return []  # Return list of results instead of yielding


    def process_data(self,
        data,
    ):
        if self.args.seq_len % 2 == 0:
            raise ValueError("kmer_len must be odd")
        num_bases = (self.args.seq_len - 1) // 2
        features_list = []
        
        signal, seq_read = data
        if seq_read.mapping_quality<self.args.mapq:
            return features_list
        read_dict = dict(seq_read.tags)
        mv_table = np.asarray(read_dict["mv"][1:])
        stride = int(read_dict["mv"][0])
        num_trimmed = read_dict["ts"]
        # norm_shift = read_dict["sm"]
        # norm_scale = read_dict["sd"]
        signal_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
        norm_signals = normalize_signals(signal_trimmed, self.args.normalize_method)
        # sshift, sscale = np.mean(signal_trimmed), float(np.std(signal_trimmed))
        # if sscale == 0.0:
        #    norm_signals = signal_trimmed
        # else:
        #    norm_signals = (signal_trimmed - sshift) / sscale
        seq = seq_read.get_forward_sequence()
        signal_group = _group_signals_by_movetable_v2(norm_signals, mv_table, stride)
        tsite_locs = get_refloc_of_methysite_in_motif(seq, set(self.motif_seqs), self.args.mod_loc)
        strand = "."
        q_to_r_poss = None
        if not seq_read.is_unmapped:
            strand = "-" if seq_read.is_reverse else "+"
            strand_code = -1 if seq_read.is_reverse else 1
            ref_start = seq_read.reference_start
            ref_end = seq_read.reference_end
            cigar_tuples = seq_read.cigartuples
            qalign_start = seq_read.query_alignment_start
            qalign_end = seq_read.query_alignment_end
            if (qalign_end-qalign_start)/seq_read.query_length<self.args.coverage_ratio:
                return features_list
            if seq_read.is_reverse:
                seq_start = len(seq) - qalign_end
                seq_end = len(seq) - qalign_start
            else:
                seq_start = qalign_start
                seq_end = qalign_end
            q_to_r_poss = get_q2tloc_from_cigar(
                cigar_tuples, strand_code, (seq_end - seq_start)
            )
        if seq_read.reference_name is None:
            ref_name = "."
        else:
            ref_name = seq_read.reference_name
        for loc_in_read in tsite_locs:
            if num_bases <= loc_in_read < len(seq) - num_bases:
                ref_pos = -1
                if not seq_read.is_unmapped:
                    if seq_start <= loc_in_read < seq_end:
                        offset_idx = loc_in_read - seq_start
                        if q_to_r_poss[offset_idx] != -1:
                            if strand == "-":
                                # pos = '.'#loc_in_read
                                ref_pos = ref_end - 1 - q_to_r_poss[offset_idx]
                            else:
                                # pos = loc_in_read
                                ref_pos = ref_start + q_to_r_poss[offset_idx]
                    else:
                        continue

                if (self.positions is not None) and (
                    key_sep.join([ref_name, str(ref_pos), strand]) not in self.positions
                ):
                    continue

                k_mer = seq[(loc_in_read - num_bases) : (loc_in_read + num_bases + 1)]
                k_seq=[base2code_dna[x] for x in k_mer]
                k_signals = signal_group[
                    (loc_in_read - num_bases) : (loc_in_read + num_bases + 1)
                ]

                signal_lens = [len(x) for x in k_signals]
                # if sum(signal_lens) > MAX_LEGAL_SIGNAL_NUM:
                #     continue

                signal_means = [np.mean(x) for x in k_signals]
                signal_stds = [np.std(x) for x in k_signals]
                sampleinfo=(
                    "\t".join(
                        [ref_name, str(ref_pos), strand, ".", seq_read.query_name,'.' ]
                    )
                )
                features_list.append((
                    sampleinfo,
                    k_seq,
                    signal_means,
                    signal_stds,
                    signal_lens,
                    _get_signals_rect(k_signals, self.args.signal_len),
                    self.args.methy_label,)
                )
        return features_list#to_tensor(features_list,device) if len(features_list) > 0 else []


class Fast5Dataset(IterableDataset):
    def __init__(self, file_list, motif_seqs, positions,device,args,chrom2len,read_strand,contig,aligner=None):
        self.files = file_list
        self.motif_seqs = motif_seqs
        self.positions = positions
        self.args = args
        self.device = device
        self.chrom2len=chrom2len
        self.read_strand = read_strand
        self.contig=contig
        self.aligner=aligner
        self.map_thr_buf = mappy.ThreadBuffer()
        

    def __iter__(self):
        # Fill the queue with Fast5 files
        fast5s_q = []
        fill_files_queue(fast5s_q, self.files)
        LOGGER.info("extract_features process-{} starts".format(os.getpid()))

        for files in fast5s_q:
            for file in files:
                # Iterate over Fast5 files
                raw_data = self.process_fast5_file(file)
                if raw_data:
                    for item in raw_data:
                        if len(item) == 0:
                            continue
                        yield to_tensor(item, self.device)
    def process_fast5_file(self, file):
        """Processes a single read from a Fast5 file."""
        # Get read information
        if self.args.is_single:
            readname = ""
            success, readinfo = self._get_read_signalinfo(
                file, readname
            )
            if success == 0:
                return []
            
            # Extract the processed information
            read_id, baseseq, signal_group,  mapinfo = readinfo
            
            # Perform further processing based on mapping or non-mapping
            if self.is_mapping:
                # Mapping-specific processing
                return self.process_mapping(read_id, baseseq, signal_group, mapinfo)
            else:
                # Non-mapping-specific processing
                return self.process_non_mapping(read_id, baseseq, signal_group, mapinfo)
        else:
            features_list_batch = []
            multi_f5 = fast5_reader.MultiFast5(file)
            for readname in iter(multi_f5):
                singlef5 = multi_f5[readname]
                success, readinfo = self._get_read_signalinfo(
                    singlef5, readname
                )
                if success == 0:
                    continue
                read_id, baseseq, signal_group,  mapinfo = readinfo
                if self.is_mapping:
                    # Mapping-specific processing
                    features_list_batch.append(
                        self.process_mapping(read_id, baseseq, signal_group)
                    )
                else:
                    # Non-mapping-specific processing
                    features_list_batch.append(
                        self.process_non_mapping(read_id, baseseq, signal_group, mapinfo)
                    )

            return features_list_batch
        
    def _convert_cigarstring2tuple(self,cigarstr):
        # tuple in (oplen, op) format, like 30M -> (30, 0)
        return [(int(m[0]), CIGAR2CODE[m[-1]]) for m in CIGAR_REGEX.findall(cigarstr)]
    def _get_read_signalinfo(self,
        fast5_fn,readname
    ):
        """

        :param fast5_fn:
        :param basecall_group:
        :param basecall_subgroup:
        :param norm_method:
        :param readname: if is_single=False, readname must not be ""
        :param is_single: is the fast5_fn is in single-read format
        :return: success(0/1), (readid, baseseq, signal_group)
        """
        try:
            success = 1

            fast5read = fast5_reader.SingleFast5(fast5_fn, self.args.is_single, readname)
            # print('1: ')
            readid = fast5read.get_readid()
            bgrp = (
                self.args.basecall_group
                if self.args.basecall_group is not None
                else fast5read.get_lastest_basecallgroup()
            )
            baseseq = fast5read.get_seq(bgrp, self.args.basecall_subgroup)
            movetable = fast5read.get_move(bgrp, self.args.basecall_subgroup)
            signal_stride = fast5read.get_stride(bgrp)
            # print('2: ')
            rawsignals = fast5read.get_raw_signal()
            rawsignals = fast5read.rescale_signals(rawsignals)
            # print('3: ')
            seggrp = fast5read.get_basecallgroup_related_sementation(bgrp)
            signal_start, signal_duration = fast5read.get_segmentation_start(
                seggrp
            ), fast5read.get_segmentation_duration(seggrp)
            # print('4: ')
            # print('5: ')
            fast5read.check_fastq_seqlen(bgrp, self.args.basecall_subgroup)
            fast5read.check_signallen_against_segmentation(seggrp)

            mapinfo = None
            if not self.args.is_mapping:
                mapinfo = fast5read.get_map_info(self.args.corrected_group, self.args.basecall_subgroup)

            fast5read.close()

            # map signal to base
            mapsignals = normalize_signals(
                rawsignals[signal_start : (signal_start + signal_duration)], self.args.normalize_method
            )
            signal_group = _group_signals_by_movetable_v2(
                mapsignals, movetable, signal_stride
            )

            return success, (readid, baseseq, signal_group,  mapinfo)
        except IOError:
            LOGGER.error("Error in reading file-{}, skipping".format(fast5_fn))
            return 0, None
        except AssertionError:
            LOGGER.error("Error in mapping signal2base, file-{}, skipping".format(fast5_fn))
            return 0, None
        except KeyError:
            LOGGER.error("Error in getting group from file-{}, skipping".format(fast5_fn))
            return 0, None
    def _compute_pct_identity(self, cigar):
        nalign, nmatch = 0, 0
        for op_len, op in cigar:
            if op not in (4, 5):
                nalign += op_len
            if op in (0, 7):
                nmatch += op_len
        return nmatch / float(nalign)
    def parse_cigar(self, r_cigar, strand, ref_len):
        fill_invalid = -1
        # get each base calls genomic position
        r_to_q_poss = np.full(ref_len + 1, fill_invalid, dtype=np.int32)
        # process cigar ops in read direction
        curr_r_pos, curr_q_pos = 0, 0
        cigar_ops = r_cigar if strand == 1 else r_cigar[::-1]
        for op_len, op in cigar_ops:
            if op == 1:
                # inserted bases into ref
                curr_q_pos += op_len
            elif op in (2, 3):
                # deleted ref bases
                for r_pos in range(curr_r_pos, curr_r_pos + op_len):
                    r_to_q_poss[r_pos] = curr_q_pos
                curr_r_pos += op_len
            elif op in (0, 7, 8):
                # aligned bases
                for op_offset in range(op_len):
                    r_to_q_poss[curr_r_pos + op_offset] = curr_q_pos + op_offset
                curr_q_pos += op_len
                curr_r_pos += op_len
            elif op == 6:
                # padding (shouldn't happen in mappy)
                pass
        r_to_q_poss[curr_r_pos] = curr_q_pos
        if r_to_q_poss[-1] == fill_invalid:
            raise ValueError(
                (
                    "Invalid cigar string encountered. Reference length: {}  Cigar "
                    + "implied reference length: {}"
                ).format(ref_len, curr_r_pos)
            )

        return r_to_q_poss
    def align_read(self, q_seq, aligner, map_thr_buf, read_id=None):
        try:
            # enumerate all alignments to avoid memory leak from mappy
            r_algn = list(aligner.map(str(q_seq), buf=map_thr_buf))[0]
        except IndexError:
            # alignment not produced
            return None

        ref_seq = str(aligner.seq(r_algn.ctg, r_algn.r_st, r_algn.r_en)).upper()
        if r_algn.strand == -1:
            try:
                ref_seq = complement_seq(ref_seq)
            except KeyError:
                LOGGER.debug("ref_seq contions U base: {}".format(ref_seq))
                ref_seq = complement_seq(ref_seq, "RNA")
        # coord 0-based
        return MAP_RES(
            read_id=read_id,
            q_seq=str(q_seq).upper(),
            ref_seq=ref_seq,
            ctg=r_algn.ctg,
            strand=r_algn.strand,
            r_st=r_algn.r_st,
            r_en=r_algn.r_en,
            q_st=r_algn.q_st,
            q_en=r_algn.q_en,
            cigar=r_algn.cigar,
            mapq=r_algn.mapq,
        )
    def _map_read_to_ref_process(self,aligner, map_conn):
        read_id, q_seq = map_conn
        
        map_res = self.align_read(q_seq, aligner,self.map_thr_buf, read_id)
        if map_res is None:
            return((0, None))
        else:
            map_res = tuple(map_res)
            return((1, map_res))

    def _extract_features(self,
        ref_mapinfo,
        chrom2len,
        positions,
        read_strand,
    ):
        # read_id, ref_seq, chrom, strand, r_st, r_en, ref_signal_grp, ref_baseprobs = ref_mapinfo
        # strand = "+" if strand == 1 else "-"

        # seg_mapping: (len(ref_seq), item_chrom, strand_code, item_ref_s, item_ref_e)
        read_id, ref_seq, ref_readlocs, ref_signal_grp, seg_mapping = (
            ref_mapinfo
        )

        if self.args.seq_len % 2 == 0:
            raise ValueError("--seq_len must be odd")
        num_bases = (self.args.seq_len - 1) // 2

        feature_lists = []
        # WARN: cannot make sure the ref_offsets are all targeted motifs in corresponding read,
        # WARN: cause there may be mismatches/indels in those postions.
        # WARN: see also parse_cigar()/_process_read_map()/_process_read_nomap()
        ref_offsets = get_refloc_of_methysite_in_motif(ref_seq, set(self.motif_seqs), self.args.mod_loc)
        for off_loc_i in range(len(ref_offsets)):
            off_loc_b = ref_offsets[off_loc_i - 1] if off_loc_i != 0 else -1
            off_loc = ref_offsets[off_loc_i]
            off_loc_a = (
                ref_offsets[off_loc_i + 1] if off_loc_i != len(ref_offsets) - 1 else -1
            )
            if num_bases <= off_loc < len(ref_seq) - num_bases:
                chrom, strand, r_st, r_en = None, None, None, None
                seg_off_loc = None
                seg_len_accum = 0
                for seg_tmp in seg_mapping:
                    seg_len_accum += seg_tmp[0]
                    if off_loc < seg_len_accum:
                        seg_off_loc = off_loc - (seg_len_accum - seg_tmp[0])
                        chrom, strand, r_st, r_en = seg_tmp[1:]
                        strand = "+" if strand == 1 else "-"
                        break

                abs_loc = r_st + seg_off_loc if strand == "+" else r_en - 1 - seg_off_loc
                read_loc = ref_readlocs[off_loc]
                tag = 0
                if off_loc_b != -1:
                    read_loc_b = ref_readlocs[off_loc_b]
                    if abs(read_loc - read_loc_b) <= 10:
                        tag = 1
                if off_loc_a != -1:
                    read_loc_a = ref_readlocs[off_loc_a]
                    if abs(read_loc_a - read_loc) <= 10:
                        tag = 1

                if (positions is not None) and (
                    key_sep.join([chrom, str(abs_loc), strand]) not in positions
                ):
                    continue

                loc_in_strand = (
                    abs_loc
                    if (strand == "+" or chrom == "read")
                    else chrom2len[chrom] - 1 - abs_loc
                )
                k_mer = ref_seq[(off_loc - num_bases) : (off_loc + num_bases + 1)]
                k_signals = ref_signal_grp[
                    (off_loc - num_bases) : (off_loc + num_bases + 1)
                ]

                signal_lens = [len(x) for x in k_signals]

                signal_means = [np.mean(x) for x in k_signals]
                signal_stds = [np.std(x) for x in k_signals]

                k_signals_rect = _get_signals_rect(k_signals, self.args.signal_len, self.args.pad_only_r)
                # gc_contont = 0
                # gc_set = {"G", "C"}
                # for i in range(max(0, off_loc - 50), min(len(ref_seq), off_loc + 50)):
                #     if ref_seq[i] in gc_set:
                #         gc_contont += 1
                sampleinfo = (
                    "\t".join(
                        [chrom,
                        abs_loc,
                        strand,
                        loc_in_strand,
                        read_id,
                        read_loc,]
                    )
                )
                feature_lists.append(
                    (
                        sampleinfo,
                        k_mer,
                        signal_means,
                        signal_stds,
                        signal_lens,
                        k_signals_rect,
                        self.args.methy_label,
                        # tag,
                        # gc_contont / 100.0,
                    )
                )
        return feature_lists
    def process_mapping(self, read_id, baseseq, signal_group):
        """Process the read when mapping information is available."""
        # Insert the alignment and mapping logic here
        # Return a tuple or dict with the processed data
        success, map_res = self._map_read_to_ref_process(self.aligner, (read_id, baseseq))
        if success == 0:
            return []
        map_res = MAP_RES(*map_res)
        if map_res.mapq < self.args.mapq:
            LOGGER.debug("mapq too low: {}, mapq: {}".format(map_res.read_id, map_res.mapq))
            return []
        if self._compute_pct_identity(map_res.cigar) < self.args.identity:
            LOGGER.debug(
                "identity too low: {}, identity: {}".format(
                    map_res.read_id, self._compute_pct_identity(map_res.cigar)
                )
            )
            return []
        if (map_res.q_en - map_res.q_st) / float(len(map_res.q_seq)) < self.args.coverage_ratio:
            return []
        try:
            r_to_q_poss = self.parse_cigar(map_res.cigar, map_res.strand, len(map_res.ref_seq))
        except Exception:
            LOGGER.debug("cigar parsing error: {}".format(map_res.read_id))
            return []

        ref_signal_grp = [
            None,
        ] * len(map_res.ref_seq)
        ref_readlocs = [
            0,
        ] * len(map_res.ref_seq)
        for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
            # signal groups
            ref_signal_grp[ref_pos] = signal_group[q_pos + map_res.q_st]
            ref_readlocs[ref_pos] = q_pos + map_res.q_st
            # trace
            try:
                base_idx = base2code_dna[map_res.ref_seq[ref_pos]]
            except Exception:
                # LOGGER.debug("error when extracting trace feature: {}-{}".format(map_res.read_id,
                #                                                                 q_pos))
                continue

        # ref_mapinfo = (read_id, map_res.ref_seq, map_res.ctg, map_res.strand, map_res.r_st, map_res.r_en,
        #                ref_signal_grp, ref_baseprobs)
        ref_mapinfo = (
            read_id,
            map_res.ref_seq,
            ref_readlocs,
            ref_signal_grp,
            [
                (
                    len(map_res.ref_seq),
                    map_res.ctg,
                    map_res.strand,
                    map_res.r_st,
                    map_res.r_en,
                ),
            ],
        )
        features_list = self._extract_features(
            ref_mapinfo,
            self.chrom2len,
            self.positions,
            self.read_strand,
        )
        return features_list
    
    def process_non_mapping(self, read_id, baseseq, signal_group, mapinfo):
        """Process the read when no mapping information is available."""
        # Insert the non-mapping processing logic here
        # Return a tuple or dict with the processed data
        if len(mapinfo) == 0:
            return []
        failed_cnt = 0

        combed_ref_seq = ""
        combed_ref_signal_grp = []
        combed_ref_baseprobs = []
        combed_ref_readlocs = []
        combed_cigartuples = []
        seg_mapping = []
        for map_idx in range(len(mapinfo)):
            (
                item_cigar,
                item_chrom,
                item_strand,
                item_read_s,
                item_read_e,
                item_ref_s,
                item_ref_e,
            ) = mapinfo[map_idx]
            cigartuple = self.__convert_cigarstring2tuple(item_cigar)
            # WARN: remove D(eltion, encoded as 2) at the end of the alignment?
            if map_idx == len(mapinfo) - 1:
                if item_strand == "+" and cigartuple[-1][-1] == 2:
                    oplen_tmp = cigartuple[-1][0]
                    cigartuple = cigartuple[:-1]
                    item_ref_e -= oplen_tmp
                if item_strand == "-" and cigartuple[0][-1] == 2:
                    oplen_tmp = cigartuple[0][0]
                    cigartuple = cigartuple[1:]
                    item_ref_s += oplen_tmp
            try:
                if item_chrom == "read":
                    assert item_strand == "+"
                    ref_seq = baseseq[item_ref_s:item_ref_e]
                    strand_code = 1
                    r_to_q_poss = self.parse_cigar(cigartuple, strand_code, len(ref_seq))
                else:
                    ref_seq = self.contig[item_chrom][item_ref_s:item_ref_e]
                    if item_strand == "-":
                        ref_seq = complement_seq(ref_seq)
                    strand_code = 0 if item_strand == "-" else 1
                    r_to_q_poss = self.parse_cigar(cigartuple, strand_code, len(ref_seq))
            except KeyError:
                LOGGER.debug(
                    "no chrom-{} in reference genome: {}".format(item_chrom, read_id)
                )
                failed_cnt += 1
                continue
            except ValueError:
                LOGGER.debug("cigar parsing error: {}".format(read_id))
                failed_cnt += 1
                continue

            ref_signal_grp = [
                None,
            ] * len(ref_seq)
            ref_readlocs = [
                0,
            ] * len(ref_seq)
            for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
                # signal groups
                ref_readlocs[ref_pos] = q_pos + item_read_s
                ref_signal_grp[ref_pos] = signal_group[q_pos + item_read_s]
                # trace
                try:
                    base_idx = base2code_dna[ref_seq[ref_pos]]
                except Exception:
                    continue

            combed_ref_seq += ref_seq
            combed_ref_readlocs += ref_readlocs
            combed_ref_signal_grp += ref_signal_grp
            combed_cigartuples += cigartuple
            seg_mapping.append(
                (len(ref_seq), item_chrom, strand_code, item_ref_s, item_ref_e)
            )
        if failed_cnt > 0:
            return []
        try:
            if self._compute_pct_identity(combed_cigartuples) < self.argsidentity:
                LOGGER.debug(
                    "identity too low: {}, identity: {}".format(
                        read_id, self._compute_pct_identity(combed_cigartuples)
                    )
                )
                return []
        except ZeroDivisionError:
            raise ZeroDivisionError("{}, {}".format(read_id, combed_cigartuples))
        del combed_cigartuples
        ref_mapinfo = (
            read_id,
            combed_ref_seq,
            combed_ref_readlocs,
            combed_ref_signal_grp,
            seg_mapping,
        )
        features_list = self._extract_features(
            ref_mapinfo,
            self.motif_seqs,
            self.chrom2len,
            self.positions,
            self.read_strand,
            self.args.mod_loc,
            self.args.seq_len,
            self.args.signal_len,
            self.args.methy_label,
            self.args.pad_only_r,
        )
        return features_list


class TsvDataset(IterableDataset):
    def __init__(self, features_file):
        """
        Dataset for reading features from a TSV file.
        
        Args:
            features_file (str): Path to the TSV file.
        """
        self.features_file = features_file

    def parse_tsv_line(self, line):
        """
        Parse a line from the TSV file.
        
        Args:
            line (str): A line from the TSV file.
            
        Returns:
            Tuple containing parsed sampleinfo, kmers, base_means, base_stds,
            base_signal_lens, k_signals, and label.
        """
        words = line.strip().split("\t")
        sampleinfo = "\t".join(words[0:6])
        kmers = [base2code_dna[x] for x in words[6]]
        base_means = [float(x) for x in words[7].split(",")]
        base_stds = [float(x) for x in words[8].split(",")]
        base_signal_lens = [int(x) for x in words[9].split(",")]
        k_signals = [[float(y) for y in x.split(",")] for x in words[10].split(";")]
        label = int(words[11])
        return sampleinfo, kmers, base_means, base_stds, base_signal_lens, k_signals, label

    def __iter__(self):
        """
        Iterate over the TSV file and yield each parsed line.
        """
        if self.features_file.endswith(".gz"):
            infile = gzip.open(self.features_file, "rt")
        else:
            infile = open(self.features_file, "r")

        for line in infile:
            # Parse line and yield the parsed result
            parsed_data = self.parse_tsv_line(line)
            yield to_tensor(parsed_data)

        infile.close()

def to_tensor(data,device='cpu'):
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, k_signals, labels = data
    b_kmers=torch.tensor(kmers, dtype=torch.float)
    b_base_means=torch.tensor(base_means, dtype=torch.float)
    b_base_stds=torch.tensor(base_stds, dtype=torch.float)
    b_base_signal_lens=torch.tensor(base_signal_lens, dtype=torch.float)
    b_k_signals=torch.tensor(k_signals, dtype=torch.float)
    #return (kmers,1,1)
    #labels = np.reshape(labels, (len(labels)))
    return sampleinfo, b_kmers, b_base_means, b_base_stds, b_base_signal_lens, b_k_signals, labels#[item for item in zip(sampleinfo, b_kmers, b_base_means, b_base_stds, b_base_signal_lens, b_k_signals, labels)]

def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    total_files = len(dataset.files)
    per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))

    start = worker_id * per_worker
    end = min(start + per_worker, total_files)
    dataset.files = dataset.files[start:end]

def collate_fn_inference(batch):
    # ref_name, ref_pos, strand, placeholder1, read_name, placeholder2, k_mer, signal_means, signal_stds, signal_lens, k_signals_rect, methy_label = zip(
    #     *batch)
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, k_signals, labels = zip(
        *batch
    )
    labels = np.reshape(labels, (len(labels)))
    # return ref_name, ref_pos, strand, placeholder1, read_name, placeholder2, k_mer, signal_means, signal_stds, signal_lens, k_signals_rect, methy_label
    return sampleinfo, torch.stack(kmers,0), torch.stack(base_means,0), torch.stack(base_stds,0), \
        torch.stack(base_signal_lens,0), torch.stack(k_signals,0), labels

def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    insertion: -1, deletion: -2, mismatch: -3
    :param r_cigar_tuple: pysam.alignmentSegment.cigartuples
    :param strand: 1/-1 for fwd/rev
    :param seq_len: read alignment length
    :return: query pos to ref pos
    """
    fill_invalid = -2
    # get each base calls genomic position
    q_to_r_poss = np.full(seq_len + 1, fill_invalid, dtype=np.int32)
    # process cigar ops in read direction
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar_tuple if strand == 1 else r_cigar_tuple[::-1]
    for op, op_len in cigar_ops:
        if op == 1:
            # inserted bases into ref
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = -1
            curr_q_pos += op_len
        elif op in (2, 3):
            # deleted ref bases
            curr_r_pos += op_len
        elif op in (0, 7, 8):
            # aligned bases
            for op_offset in range(op_len):
                q_to_r_poss[curr_q_pos + op_offset] = curr_r_pos + op_offset
            curr_q_pos += op_len
            curr_r_pos += op_len
        elif op == 6:
            # padding (shouldn't happen in mappy)
            pass
    q_to_r_poss[curr_q_pos] = curr_r_pos
    if q_to_r_poss[-1] == fill_invalid:
        raise ValueError(
            (
                "Invalid cigar string encountered. Reference length: {}  Cigar "
                + "implied reference length: {}"
            ).format(seq_len, curr_r_pos)
        )
    return q_to_r_poss

def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """

    :param trimed_signals:
    :param movetable: numpy array
    :param stride:
    :return:
    """
    assert movetable[0] == 1
    # TODO: signal_duration not exactly equal to call_events * stride
    # TODO: maybe this move * stride way is not right!
    assert len(trimed_signals) >= len(movetable) * stride
    signal_group = []
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    for move_idx in range(len(move_pos) - 1):
        start, end = move_pos[move_idx], move_pos[move_idx + 1]
        signal_group.append(trimed_signals[(start * stride) : (end * stride)].tolist())
    assert len(signal_group) == np.sum(movetable)
    return signal_group

def _get_signals_rect(signals_list, signals_len=16, pad_only_r=False):
    signals_rect = []
    for signals_tmp in signals_list:
        signals = list(np.around(signals_tmp, decimals=6))
        if len(signals) < signals_len:
            pad0_len = signals_len - len(signals)
            if not pad_only_r:
                pad0_left = pad0_len // 2
                pad0_right = pad0_len - pad0_left
                signals = [0.0] * pad0_left + signals + [0.0] * pad0_right
            else:
                signals = signals + [0.0] * pad0_len
        elif len(signals) > signals_len:
            signals = [
                signals[x]
                for x in sorted(random.sample(range(len(signals)), signals_len))
            ]
        signals_rect.append(signals)
    return signals_rect