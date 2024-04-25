"""the feature extraction module.
output format:
chrom, pos, alignstrand, pos_in_strand, readname, read_loc, k_mer, signal_means,
signal_stds, signal_lens, base_probs, raw_signals, methy_label
"""
# TODO: use pyguppyclient (https://github.com/nanoporetech/pyguppyclient) instead of --fast5-out

from __future__ import absolute_import

import sys
import os
import argparse
import time
# import h5py
import random
import numpy as np
import multiprocessing as mp
# TODO: when using below import, will raise AttributeError: 'Queue' object has no attribute '_size'
# TODO: in call_mods module, didn't figure out why
# from .utils.process_utils import Queue
from multiprocessing import Queue
from statsmodels import robust

from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import get_fast5s
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import complement_seq

from .utils.ref_reader import get_contig2len
from .utils.ref_reader import get_contig2len_n_seq


from .utils.process_utils import base2code_dna
from .utils import bam_reader

from .utils.process_utils import CIGAR_REGEX
from .utils.process_utils import CIGAR2CODE

import mappy
import threading
from collections import namedtuple
from .utils.process_utils import get_logger
import pod5
import pysam
from .extract_features import _group_signals_by_movetable_v2
from .extract_features import _get_signals_rect
from .extract_features import _write_featurestr
from .utils.process_utils import split_list
from .utils.process_utils import _read_position_file

LOGGER = get_logger(__name__)

queue_size_border = 2000
time_wait = 3

key_sep = "||"

MAP_RES = namedtuple('MAP_RES', (
    'read_id', 'q_seq', 'ref_seq', 'ctg', 'strand', 'r_st', 'r_en',
    'q_st', 'q_en', 'cigar', 'mapq'))



################utils###################
def _read_position_file(position_file):
    postions = set()
    with open(position_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            postions.add(key_sep.join(words[:3]))
    return postions

def _features_to_str(features):
    """
    :param features: a tuple
    :return:
    """
    chrom, pos, alignstrand, loc_in_ref, readname, read_loc, k_mer, signal_means, signal_stds, \
        signal_lens, k_signals_rect, methy_label = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    #base_probs_text = ",".join([str(x) for x in np.around(k_baseprobs, decimals=6)])
    k_signals_text = ';'.join([",".join([str(y) for y in x]) for x in k_signals_rect])
    #freq_text = ';'.join([",".join([str(y) for y in x]) for x in freq])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, str(read_loc), k_mer, means_text,
                      stds_text, signal_len_text, k_signals_text, str(methy_label)])

def process_data(data,motif_seqs,positions,kmer_len,signals_len,methyloc=0,methy_label=1):
    if kmer_len % 2 == 0:
        raise ValueError("kmer_len must be odd")
    num_bases = (kmer_len - 1) // 2
    features_list = []
    signal,seq_read=data
    read_dict=dict(seq_read.tags)
    mv_table=np.asarray(read_dict['mv'][1:])
    stride=int(read_dict['mv'][0])
    num_trimmed=read_dict["ts"]
    norm_shift = read_dict["sm"]
    norm_scale = read_dict["sd"]
    if num_trimmed >= 0:
        signal_trimmed = (signal[num_trimmed:] - norm_shift) / norm_scale
    else:
        signal_trimmed = (signal[:num_trimmed] - norm_shift) / norm_scale
    sshift, sscale = np.mean(signal_trimmed), float(np.std(signal_trimmed))
    if sscale == 0.0:
        norm_signals = signal_trimmed
    else:
        norm_signals = (signal_trimmed - sshift) / sscale
    seq=seq_read.get_forward_sequence()
    signal_group = _group_signals_by_movetable_v2(norm_signals, mv_table, stride)
    tsite_locs = get_refloc_of_methysite_in_motif(seq, set(motif_seqs), methyloc)
    strand="-" if seq_read.is_reverse else '+'
    ref_start=seq_read.reference_start
    
    for loc_in_read in tsite_locs:
        if num_bases <= loc_in_read < len(seq) - num_bases:
            if strand == '-':
                pos = ref_start + len(seq) - 1 - loc_in_read
            else:
                pos = ref_start + loc_in_read
            k_mer = seq[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]
            #k_seq=[base2code_dna[x] for x in k_mer]
            k_signals = signal_group[(loc_in_read - num_bases):(loc_in_read + num_bases + 1)]

            signal_lens = [len(x) for x in k_signals]
            # if sum(signal_lens) > MAX_LEGAL_SIGNAL_NUM:
            #     continue

            signal_means = [np.mean(x) for x in k_signals]
            signal_stds = [np.std(x) for x in k_signals]
            if seq_read.reference_name is None:
                ref_name='.'
            else:
                ref_name=seq_read.reference_name
            #sampleinfo='\t'.join([ref_name,str(pos) , 't', '.', seq_read.query_name, strand])
            if (positions is not None) and (key_sep.join([ref_name,str(pos), strand]) not in positions):
                continue
            
            features_list.append(_features_to_str((ref_name,str(pos) , strand, '.', seq_read.query_name, '.',k_mer, signal_means, signal_stds,signal_lens,
                                    _get_signals_rect(k_signals, signals_len),methy_label)))
    return features_list
def process_sig_seq(seq_index,sig_dr,feature_Q,motif_seqs,positions,kmer_len,signals_len,qsize_limit=4,time_wait=1):
    chunk=[]
    for filename in sig_dr:
        with pod5.Reader(filename) as reader:
            for read_record in reader.reads():
                if feature_Q.qsize()>qsize_limit:
                    time.sleep(time_wait)
                read_name=str(read_record.read_id)
                signal=read_record.signal
                if signal is None:
                    continue
                try:
                    for seq_read in seq_index.get_alignments(read_name):
                        seq = seq_read.get_forward_sequence()
                        if seq is None:
                            continue
                        data=(signal,seq_read)
                        feature_lists=process_data(data,motif_seqs,positions,kmer_len,signals_len)
                        #chunk.append(feature_lists)
                        #if len(chunk)>=r_batch_size:
                        #print(len(feature_lists[0]),flush=True)
                        feature_Q.put(feature_lists)
                        chunk=[]               
                except KeyError:
                    print('Read:%s not found in BAM file' %read_name, flush=True)
                    continue
    if len(chunk)>0:
        feature_Q.put(chunk)
        chunk=[]

def extract_features(args):
    start = time.time()
    if args.reference_path is not None:
        ref_path = os.path.abspath(args.reference_path)
        if not os.path.exists(ref_path):
            raise ValueError("--reference_path not set right!")
    else:
        ref_path = None
    

    LOGGER.info("[extract] starts")

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(input_dir):
        raise ValueError("--input_dir not set right!")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError("--input_dir not a directory")
    LOGGER.info("read position file if it is not None")
    positions = None
    if args.positions is not None:
        positions = _read_position_file(args.positions)
    is_recursive = str2bool(args.recursively)
    is_dna = False if args.rna else True
    motif_seqs = get_motif_seqs(args.motifs, is_dna)
    bam_index=bam_reader.ReadIndexedBam(args.bam)
    pod5_dr=[]
    for pod5_name in os.listdir(input_dir):
        if pod5_name.endswith('.pod5'):
            pod5_path = '/'.join([input_dir, pod5_name])
            pod5_dr.append(pod5_path)
    #pod5_dr=pod5.DatasetReader(input_dir, recursive=is_recursive)
    features_batch_q = Queue()
    error_q = Queue()
    p_rfs=[]

    nproc = args.nproc-1
    if nproc<len(pod5_dr):
        data=split_list(pod5_dr, nproc)
        for sig_data in data:
            p_rf = mp.Process(target=process_sig_seq, args=(bam_index,sig_data, features_batch_q,motif_seqs,positions,args.seq_len,args.signal_len,
                                                            args.r_batch_size),
                          name="reader")
            p_rf.daemon = True
            p_rf.start()
            p_rfs.append(p_rf)
    else:
        data=split_list(pod5_dr, len(pod5_dr))
        for sig_data in data:
            p_rf = mp.Process(target=process_sig_seq, args=(bam_index,sig_data, features_batch_q,motif_seqs,positions,args.seq_len,args.signal_len,
                                                            args.r_batch_size),
                          name="reader")
            p_rf.daemon = True
            p_rf.start()
            p_rfs.append(p_rf)
    #if nproc < 2:
    #    nproc = 2

    p_w = mp.Process(target=_write_featurestr, args=(args.write_path, features_batch_q, args.w_batch_num,
                                                     str2bool(args.w_is_dir)),
                     name="writer")
    p_w.daemon = True
    p_w.start()
    # finish processes
    for pb in p_rfs:
        pb.join()
    features_batch_q.put("kill")

    p_w.join()

    LOGGER.info("[extract] finished, cost {:.1f}s".format(time.time()-start))

def main():
    extraction_parser = argparse.ArgumentParser("extract features from pod5 for "
                                                "training or testing."
                                                "\nIt is suggested that running this module 1 flowcell a time, "
                                                "or a group of flowcells a time, "
                                                "if the whole data is extremely large.")
    ep_input = extraction_parser.add_argument_group("INPUT")
    ep_input.add_argument("--input_dir", "-i", action="store", type=str,
                          required=True,
                          help="the directory of fast5 files")
    ep_input.add_argument("--recursively", "-r", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from input_dir recursively. '
                               'default true, t, yes, 1')
    ep_input.add_argument("--reference_path", action="store",
                          type=str, required=False,default=None,
                          help="the reference file to be used, usually is a .fa file")
    ep_input.add_argument("--rna", action="store_true", default=False, required=False,
                          help='the fast5 files are from RNA samples. if is rna, the signals are reversed. '
                               'NOTE: Currently no use, waiting for further extentsion')
    ep_input.add_argument('--bam', type=str, 
                        help='the bam filepath')
    ep_extraction = extraction_parser.add_argument_group("EXTRACTION")
    

    ep_extraction.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                               default="mad", required=False,
                               help="the way for normalizing signals in read level. "
                                    "mad or zscore, default mad")
    ep_extraction.add_argument("--methy_label", action="store", type=int,
                               choices=[1, 0], required=False, default=1,
                               help="the label of the interested modified bases, this is for training."
                                    " 0 or 1, default 1")
    ep_extraction.add_argument("--seq_len", action="store",
                               type=int, required=False, default=21,
                               help="len of kmer. default 21")
    ep_extraction.add_argument("--signal_len", action="store",
                               type=int, required=False, default=16,
                               help="the number of signals of one base to be used in deepsignal_plant, default 16")
    ep_extraction.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    ep_extraction.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    # ep_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    ep_extraction.add_argument("--positions", action="store", type=str,
                               required=False, default=None,
                               help="file with a list of positions interested (must be formatted as tab-separated file"
                                    " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                                    "need to be set. --positions is used to narrow down the range of the trageted "
                                    "motif locs. default None")
    ep_extraction.add_argument("--r_batch_size", action="store", type=int, default=50,
                               required=False,
                               help="number of files to be processed by each process one time, default 50")
    ep_extraction.add_argument("--pad_only_r", action="store_true", default=False,
                               help="pad zeros to only the right of signals array of one base, "
                               "when the number of signals is less than --signal_len. "
                               "default False (pad in two sides).")

    ep_mape = extraction_parser.add_argument_group("MAPe")

    ep_mapping = extraction_parser.add_argument_group("MAPPING")
    ep_mapping.add_argument("--mapping", action="store_true", default=False, required=False,
                            help='use MAPPING to get alignment, default false')
    ep_mapping.add_argument("--mapq", type=int, default=10, required=False,
                            help="MAPping Quality cutoff for selecting alignment items, default 10")
    ep_mapping.add_argument("--identity", type=float, default=0.75, required=False,
                            help="identity cutoff for selecting alignment items, default 0.75")
    ep_mapping.add_argument("--coverage_ratio", type=float, default=0.75, required=False,
                            help="percent of coverage, read alignment len against read len, default 0.75")
    ep_mapping.add_argument("--best_n", "-n", type=int, default=1, required=False,
                            help="best_n arg in mappy(minimap2), default 1")

    ep_output = extraction_parser.add_argument_group("OUTPUT")
    ep_output.add_argument("--write_path", "-o", action="store",
                           type=str, required=True,
                           help='file path to save the features')
    ep_output.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    ep_output.add_argument("--w_batch_num", action="store",
                           type=int, required=False, default=200,
                           help='features batch num to save in a single writed file when --is_dir is true')

    extraction_parser.add_argument("--nproc", "-p", action="store", type=int, default=10,
                                   required=False,
                                   help="number of processes to be used, default 10")

    extraction_args = extraction_parser.parse_args()
    display_args(extraction_args)

    extract_features(extraction_args)
if __name__ == '__main__':
    sys.exit(main())
