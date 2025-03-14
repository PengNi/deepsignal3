#!/usr/bin/python
from __future__ import absolute_import

import sys
import argparse

from .utils.process_utils import display_args,detect_file_type,validate_path

from ._version import DEEPSIGNAL3_VERSION


def main_extraction(args):
    from .extract_features import extract_features
    from .extract_features_pod5 import extract_features as extract_features_pod5
    
    display_args(args)
    input_path = validate_path(args.input_path, "--input_path")
    file_type = detect_file_type(input_path)
    if file_type in ['pod5', 'slow5']:
        extract_features_pod5(args)
    else:
        extract_features(args)


def main_call_mods(args):
    from .call_modifications import call_mods

    # from .call_modifications_transfer import call_mods as call_mods_transfer
    # from .call_modifications_domain import call_mods as call_mods_domain
    # from .call_modifications_cg import call_mods as call_mods_cg
    # from .call_modifications_cg_combine import call_mods as call_mods_cg_combine
    # from .call_modifications_freq import call_mods as call_mods_freq

    display_args(args)
    if args.transfer:
        print("transfer")
        # call_mods_transfer(args)
    elif args.domain:
        print("domain")
        # call_mods_domain(args)
    elif args.freq:
        print("freq")
        # call_mods_freq(args)
    else:
        call_mods(args)


def main_call_freq(args):
    from .call_mods_freq import call_mods_frequency_to_file

    display_args(args)
    call_mods_frequency_to_file(args)


def main_train(args):
    from .train import (
        train,
        train_transfer,
        train_domain,
        train_fusion,
        train_cnn,
        train_cg,
        train_combine,
        trainFreq,
        trainFreq_mp,
    )

    display_args(args)
    if args.transfer:
        print("transfer")
        train_transfer(args)
    elif args.domain:
        print("domain")
        train_domain(args)
    elif args.freq:
        print("freq")
        trainFreq_mp(args)
    else:
        train(args)


def main_denoise(args):
    from .denoise import denoise
    import time

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    denoise(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


def main_trainm(args):
    from .train_multigpu import train_multigpu

    display_args(args)
    train_multigpu(args)


def main():
    parser = argparse.ArgumentParser(
        prog="deepsignal3",
        description="deepsignal3 detects base modifications from Nanopore "
        "r10.4 reads, which contains the following modules:\n"
        "\t%(prog)s call_mods: call modifications\n"
        "\t%(prog)s call_freq: call frequency of modifications "
        "at genome level\n"
        "\t%(prog)s extract: extract features from corrected (tombo) "
        "fast5s for training or testing\n"
        "\t%(prog)s train: train a model, need two independent "
        "datasets for training and validating\n"
        "\t%(prog)s trainm: train multigpu\n"
        # "\t%(prog)s denoise: denoise training samples by deep-learning, "
        # "filter false positive samples (and false negative samples)"
        ,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="deepsignal3 version: {}".format(DEEPSIGNAL3_VERSION),
        help="show deepsignal3 version and exit.",
    )

    subparsers = parser.add_subparsers(
        title="modules", help="deepsignal3 modules, use -h/--help for help"
    )
    sub_call_mods = subparsers.add_parser("call_mods", description="call modifications")
    sub_call_freq = subparsers.add_parser(
        "call_freq", description="call frequency of modifications at genome level"
    )
    sub_extract = subparsers.add_parser(
        "extract",
        description="extract features from corrected (tombo) fast5s for "
        "training or testing."
        "\nIt is suggested that running this module 1 flowcell "
        "a time, or a group of flowcells a time, "
        "if the whole data is extremely large.",
    )
    sub_train = subparsers.add_parser(
        "train",
        description="train a model, need two independent datasets for training "
        "and validating",
    )
    sub_denoise = subparsers.add_parser(
        "denoise",
        description="denoise training samples by deep-learning, "
        "filter false positive samples (and "
        "false negative samples).",
    )
    sub_trainm = subparsers.add_parser("trainm", description="[EXPERIMENTAL]train a model using multi gpus")

    # sub_call_mods =============================================================================================
    sc_input = sub_call_mods.add_argument_group("INPUT")
    sc_input.add_argument(
        "--input_path",
        "-i",
        action="store",
        type=str,
        required=True,
        help="the input path, can be a signal_feature file from extract_features.py, "
        "or a directory of fast5 files. If a directory of fast5 files is provided, "
        "args in FAST5_EXTRACTION and MAPPING should (reference_path must) be provided.",
    )
    sc_input.add_argument(
        "--r_batch_size",
        action="store",
        type=int,
        default=50,
        required=False,
        help="number of files to be processed by each process one time, default 50. ONLE EFFECTIVE IN FAST5 EXTRACTION",
    )
    sc_input.add_argument("--bam", type=str, help="the bam filepath")

    sc_call = sub_call_mods.add_argument_group("CALL")
    sc_call.add_argument(
        "--model_path",
        "-m",
        action="store",
        type=str,
        required=True,
        help="file path of the trained model (.ckpt)",
    )
    sc_call.add_argument(
        "--classifier_path",
        "-f",
        action="store",
        type=str,
        help="file path of the trained classifier model (.ckpt)",
    )

    # model input
    sc_call.add_argument(
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
    )
    sc_call.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    sc_call.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="signal num of one base, default 15",
    )

    # model param
    sc_call.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    sc_call.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    sc_call.add_argument("--class_num", type=int, default=2, required=False)
    sc_call.add_argument("--dropout_rate", type=float, default=0, required=False)
    sc_call.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    sc_call.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    sc_call.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    sc_call.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    sc_call.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )

    sc_call.add_argument(
        "--batch_size",
        "-b",
        default=512,
        type=int,
        required=False,
        action="store",
        help="batch size, default 512",
    )

    # BiLSTM model param
    sc_call.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )

    sc_output = sub_call_mods.add_argument_group("OUTPUT")
    sc_output.add_argument(
        "--result_file",
        "-o",
        action="store",
        type=str,
        required=True,
        help="the file path to save the predicted result",
    )

    sc_f5 = sub_call_mods.add_argument_group("EXTRACTION")
    sc_f5.add_argument(
        "--single",
        action="store_true",
        default=False,
        required=False,
        help="the fast5 files are in single-read format",
    )
    sc_f5.add_argument(
        "--recursively",
        "-r",
        action="store",
        type=str,
        required=False,
        default="yes",
        help="is to find fast5/pod5 files from fast5 dir recursively. "
        "default true, t, yes, 1",
    )
    sc_f5.add_argument(
        "--rna",
        action="store_true",
        default=False,
        required=False,
        help="the fast5/pod5 files are from RNA samples. if is rna, the signals are reversed. "
        "NOTE: Currently no use, waiting for further extentsion",
    )
    sc_f5.add_argument(
        "--basecall_group",
        action="store",
        type=str,
        required=False,
        default=None,
        help="basecall group generated by Guppy. e.g., Basecall_1D_000",
    )
    sc_f5.add_argument(
        "--basecall_subgroup",
        action="store",
        type=str,
        required=False,
        default="BaseCalled_template",
        help="the basecall subgroup of fast5 files. default BaseCalled_template",
    )
    sc_f5.add_argument(
        "--reference_path",
        action="store",
        type=str,
        required=False,
        help="the reference file to be used, usually is a .fa file",
    )
    sc_f5.add_argument(
        "--normalize_method",
        action="store",
        type=str,
        choices=["mad", "zscore"],
        default="mad",
        required=False,
        help="the way for normalizing signals in read level. "
        "mad or zscore, default mad",
    )
    sc_f5.add_argument(
        "--methy_label",
        action="store",
        type=int,
        choices=[1, 0],
        required=False,
        default=1,
        help="the label of the interested modified bases, this is for training."
        " 0 or 1, default 1",
    )
    sc_f5.add_argument(
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
    sc_f5.add_argument(
        "--mod_loc",
        action="store",
        type=int,
        required=False,
        default=0,
        help="0-based location of the targeted base in the motif, default 0",
    )
    sc_f5.add_argument(
        "--pad_only_r",
        action="store_true",
        default=False,
        help="pad zeros to only the right of signals array of one base, "
        "when the number of signals is less than --signal_len. "
        "default False (pad in two sides).",
    )
    sc_f5.add_argument(
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

    sc_f5.add_argument(
        "--trace",
        action="store_true",
        default=False,
        required=False,
        help="use trace, default false",
    )
    sc_f5.add_argument(
        "--gc_content",
        action="store_true",
        default=False,
        required=False,
        help="extract gc content feature",
    )

    sc_mape = sub_call_mods.add_argument_group("MAPe")
    sc_mape.add_argument(
        "--corrected_group",
        action="store",
        type=str,
        required=False,
        default="RawGenomeCorrected_000",
        help="the corrected_group of fast5 files, " "default RawGenomeCorrected_000",
    )

    sc_mapping = sub_call_mods.add_argument_group("MAPPING")
    sc_mapping.add_argument(
        "--mapping",
        action="store_true",
        default=False,
        required=False,
        help="use MAPPING to get alignment, default false",
    )
    sc_mapping.add_argument(
        "--mapq",
        type=int,
        default=1,
        required=False,
        help="MAPping Quality cutoff for selecting alignment items, default 1",
    )
    sc_mapping.add_argument(
        "--identity",
        type=float,
        default=0.0,
        required=False,
        help="identity cutoff for selecting alignment items, default 0.0",
    )
    sc_mapping.add_argument(
        "--coverage_ratio",
        type=float,
        default=0.50,
        required=False,
        help="percent of coverage, read alignment len agaist read len, default 0.50",
    )
    sc_mapping.add_argument(
        "--best_n",
        "-n",
        type=int,
        default=1,
        required=False,
        help="best_n arg in mappy(minimap2), default 1",
    )
    sc_mapping.add_argument(
        "--bwa",
        action="store_true",
        default=False,
        required=False,
        help="use bwa instead of minimap2 for alignment",
    )
    sc_mapping.add_argument(
        "--path_to_bwa",
        type=str,
        default=None,
        required=False,
        help="full path to the executable binary bwa file. If not "
        "specified, it is assumed that bwa is in the PATH.",
    )

    sub_call_mods.add_argument(
        "--nproc",
        "-p",
        action="store",
        type=int,
        default=10,
        required=False,
        help="number of processes to be used, default 10.",
    )
    sub_call_mods.add_argument(
        "--nproc_gpu",
        action="store",
        type=int,
        default=2,
        required=False,
        help="number of processes to use gpu (if gpu is available), "
        "1 or a number less than nproc-1, no more than "
        "nproc/4 is suggested. default 2.",
    )

    sub_call_mods.set_defaults(func=main_call_mods)

    # sub_call_freq =====================================================================================
    scf_input = sub_call_freq.add_argument_group("INPUT")
    scf_input.add_argument(
        "--input_path",
        "-i",
        action="append",
        type=str,
        required=True,
        help="an output file from call_mods/call_modifications.py, or a directory contains "
        'a bunch of output files. this arg is in "append" mode, can be used multiple times',
    )
    scf_input.add_argument(
        "--file_uid",
        type=str,
        action="store",
        required=False,
        default=None,
        help="a unique str which all input files has, this is for finding all input files "
        "and ignoring the not-input-files in a input directory. if input_path is a file, "
        "ignore this arg.",
    )

    scf_output = sub_call_freq.add_argument_group("OUTPUT")
    scf_output.add_argument(
        "--result_file",
        "-o",
        action="store",
        type=str,
        required=True,
        help="the file path to save the result",
    )

    scf_cal = sub_call_freq.add_argument_group("CAlCULATE")
    scf_cal.add_argument(
        "--bed",
        action="store_true",
        default=False,
        help="save the result in bedMethyl format",
    )
    scf_cal.add_argument(
        "--sort", action="store_true", default=False, help="sort items in the result"
    )
    scf_cal.add_argument(
        "--prob_cf",
        type=float,
        action="store",
        required=False,
        default=0.5,
        help="this is to remove ambiguous calls. "
        "if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 "
        "means use all calls. range [0, 1], default 0.5.",
    )

    sub_call_freq.set_defaults(func=main_call_freq)

    # sub_extract ============================================================================
    se_input = sub_extract.add_argument_group("INPUT")
    se_input.add_argument(
        "--input_dir",
        "-i",
        action="store",
        type=str,
        required=True,
        help="the directory of fast5/pod5 files",
    )
    se_input.add_argument(
        "--recursively",
        "-r",
        action="store",
        type=str,
        required=False,
        default="yes",
        help="is to find fast5/pod5 files from input_dir recursively. "
        "default true, t, yes, 1",
    )
    se_input.add_argument(
        "--single",
        action="store_true",
        default=False,
        required=False,
        help="the fast5 files are in single-read format",
    )
    se_input.add_argument(
        "--reference_path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="the reference file to be used, usually is a .fa file",
    )
    se_input.add_argument(
        "--rna",
        action="store_true",
        default=False,
        required=False,
        help="the fast5/pod5 files are from RNA samples. if is rna, the signals are reversed. "
        "NOTE: Currently no use, waiting for further extentsion",
    )
    se_input.add_argument("--bam", type=str, help="the bam filepath")
    se_extraction = sub_extract.add_argument_group("EXTRACTION")
    se_extraction.add_argument(
        "--basecall_group",
        action="store",
        type=str,
        required=False,
        default=None,
        help="basecall group generated by Guppy. e.g., Basecall_1D_000",
    )
    se_extraction.add_argument(
        "--basecall_subgroup",
        action="store",
        type=str,
        required=False,
        default="BaseCalled_template",
        help="the basecall subgroup of fast5 files. default BaseCalled_template",
    )
    se_extraction.add_argument(
        "--normalize_method",
        action="store",
        type=str,
        choices=["mad", "zscore"],
        default="mad",
        required=False,
        help="the way for normalizing signals in read level. "
        "mad or zscore, default mad",
    )
    se_extraction.add_argument(
        "--methy_label",
        action="store",
        type=int,
        choices=[1, 0],
        required=False,
        default=1,
        help="the label of the interested modified bases, this is for training."
        " 0 or 1, default 1",
    )
    se_extraction.add_argument(
        "--seq_len",
        action="store",
        type=int,
        required=False,
        default=21,
        help="len of kmer. default 21",
    )
    se_extraction.add_argument(
        "--signal_len",
        action="store",
        type=int,
        required=False,
        default=15,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    se_extraction.add_argument(
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
    se_extraction.add_argument(
        "--mod_loc",
        action="store",
        type=int,
        required=False,
        default=0,
        help="0-based location of the targeted base in the motif, default 0",
    )
    # ep_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    se_extraction.add_argument(
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
    se_extraction.add_argument(
        "--r_batch_size",
        action="store",
        type=int,
        default=50,
        required=False,
        help="number of files to be processed by each process one time, default 50. ONLE EFFECTIVE IN FAST5 EXTRACTION",
    )
    se_extraction.add_argument(
        "--pad_only_r",
        action="store_true",
        default=False,
        help="pad zeros to only the right of signals array of one base, "
        "when the number of signals is less than --signal_len. "
        "default False (pad in two sides).",
    )
    se_extraction.add_argument(
        "--trace",
        action="store_true",
        default=False,
        required=False,
        help="use trace, default false",
    )

    se_mape = sub_extract.add_argument_group("MAPe")
    se_mape.add_argument(
        "--corrected_group",
        action="store",
        type=str,
        required=False,
        default="RawGenomeCorrected_000",
        help="the corrected_group of fast5 files, " "default RawGenomeCorrected_000",
    )

    se_mapping = sub_extract.add_argument_group("MAPPING")
    se_mapping.add_argument(
        "--mapping",
        action="store_true",
        default=False,
        required=False,
        help="use MAPPING to get alignment, default false",
    )
    se_mapping.add_argument(
        "--mapq",
        type=int,
        default=1,
        required=False,
        help="MAPping Quality cutoff for selecting alignment items, default 1",
    )
    se_mapping.add_argument(
        "--identity",
        type=float,
        default=0.0,
        required=False,
        help="identity cutoff for selecting alignment items, default 0.0",
    )
    se_mapping.add_argument(
        "--coverage_ratio",
        type=float,
        default=0.50,
        required=False,
        help="percent of coverage, read alignment len agaist read len, default 0.50",
    )
    se_mapping.add_argument(
        "--best_n",
        "-n",
        type=int,
        default=1,
        required=False,
        help="best_n arg in mappy(minimap2), default 1",
    )

    se_output = sub_extract.add_argument_group("OUTPUT")
    se_output.add_argument(
        "--write_path",
        "-o",
        action="store",
        type=str,
        required=True,
        help="file path to save the features",
    )
    se_output.add_argument(
        "--w_is_dir",
        action="store",
        type=str,
        required=False,
        default="no",
        help="if using a dir to save features into multiple files",
    )
    se_output.add_argument(
        "--w_batch_num",
        action="store",
        type=int,
        required=False,
        default=200,
        help="features batch num to save in a single writed file when --is_dir is true",
    )

    sub_extract.add_argument(
        "--nproc",
        "-p",
        action="store",
        type=int,
        default=10,
        required=False,
        help="number of processes to be used, default 10",
    )

    sub_extract.set_defaults(func=main_extraction)

    # sub_train =====================================================================================
    st_input = sub_train.add_argument_group("INPUT")
    st_input.add_argument("--train_file", type=str, required=True)
    st_input.add_argument("--valid_file", type=str, required=True)

    st_output = sub_train.add_argument_group("OUTPUT")
    st_output.add_argument("--model_dir", type=str, required=True)

    st_train = sub_train.add_argument_group("TRAIN")
    # model input
    st_train.add_argument(
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
    )
    st_train.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    st_train.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    # model param
    st_train.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    st_train.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    st_train.add_argument("--class_num", type=int, default=2, required=False)
    st_train.add_argument("--dropout_rate", type=float, default=0.5, required=False)
    st_train.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    st_train.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    st_train.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    st_train.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    st_train.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )
    # BiLSTM model param
    st_train.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )
    # model training
    st_train.add_argument(
        "--optim_type",
        type=str,
        default="Adam",
        choices=["Adam", "RMSprop", "SGD"],
        required=False,
        help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop', default Adam",
    )
    st_train.add_argument("--batch_size", type=int, default=512, required=False)
    st_train.add_argument("--lr", type=float, default=0.001, required=False)
    st_train.add_argument(
        "--max_epoch_num",
        action="store",
        default=15,
        type=int,
        required=False,
        help="max epoch num, default 15",
    )
    st_train.add_argument(
        "--min_epoch_num",
        action="store",
        default=5,
        type=int,
        required=False,
        help="min epoch num, default 5",
    )
    st_train.add_argument("--step_interval", type=int, default=100, required=False)

    st_train.add_argument("--pos_weight", type=float, default=1.0, required=False)
    # st_train.add_argument('--seed', type=int, default=1234,
    #                        help='random seed')
    # else
    st_train.add_argument("--tmpdir", type=str, default="/tmp", required=False)
    st_train.add_argument(
        "--transfer",
        action="store_true",
        default=False,
        help="weather use transfer learning",
    )
    st_train.add_argument(
        "--domain",
        action="store_true",
        default=False,
        help="weather use domain attribute",
    )
    st_train.add_argument(
        "--freq", action="store_true", default=False, help="weather use freq attribute"
    )

    sub_train.set_defaults(func=main_train)

    # # sub_denoise =====================================================================================
    sd_input = sub_denoise.add_argument_group("INPUT")
    sd_input.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="file containing (combined positive and "
        "negative) samples for training. better been "
        "balanced in kmer level.",
    )
    #
    sd_train = sub_denoise.add_argument_group("TRAIN")
    sd_train.add_argument(
        "--is_filter_fn",
        type=str,
        default="no",
        required=False,
        help="is filter false negative samples, , 'yes' or 'no', default no",
    )
    # # model input
    sd_train.add_argument(
        "--model_type",
        type=str,
        default="signal_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: signal_bilstm",
    )
    sd_train.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    sd_train.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    # # model param
    sd_train.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    sd_train.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    sd_train.add_argument("--class_num", type=int, default=2, required=False)
    sd_train.add_argument("--dropout_rate", type=float, default=0.5, required=False)
    sd_train.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    sd_train.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    sd_train.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    sd_train.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    sd_train.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )
    # # BiLSTM model param
    sd_train.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )
    # # model training
    sd_train.add_argument("--pos_weight", type=float, default=1.0, required=False)
    sd_train.add_argument("--batch_size", type=int, default=512, required=False)
    sd_train.add_argument("--lr", type=float, default=0.001, required=False)
    sd_train.add_argument("--epoch_num", type=int, default=3, required=False)
    sd_train.add_argument("--step_interval", type=int, default=100, required=False)
    #
    sd_denoise = sub_denoise.add_argument_group("DENOISE")
    sd_denoise.add_argument("--iterations", type=int, default=10, required=False)
    sd_denoise.add_argument("--rounds", type=int, default=3, required=False)
    sd_denoise.add_argument(
        "--score_cf",
        type=float,
        default=0.5,
        required=False,
        help="score cutoff to keep high quality (which prob>=score_cf) positive samples. "
        "usually <= 0.5, default 0.5",
    )
    sd_denoise.add_argument(
        "--kept_ratio",
        type=float,
        default=0.99,
        required=False,
        help="kept ratio of samples, to end denoise process",
    )
    sd_denoise.add_argument(
        "--fst_iter_prob",
        action="store_true",
        default=False,
        help="if output probs of samples after 1st iteration",
    )
    sd_denoise.add_argument("--nodes", default=1, type=int,
                              help="number of nodes for distributed training, default 1")
    sd_denoise.add_argument("--ngpus_per_node", default=2, type=int,
                              help="number of GPUs per node for distributed training, default 2")
    sd_denoise.add_argument("--dist-url", default="tcp://127.0.0.1:12315", type=str,
                              help="url used to set up distributed training")
    sd_denoise.add_argument("--node_rank", default=0, type=int,
                              help="node rank for distributed training, default 0")
    sd_denoise.add_argument("--epoch_sync", action="store_true", default=False,
                              help="if sync model params of gpu0 to other local gpus after per epoch")
    sd_denoise.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")
    #
    sub_denoise.set_defaults(func=main_denoise)

    # sub_train_multigpu =====================================================================================
    stm_input = sub_trainm.add_argument_group("INPUT")
    stm_input.add_argument('--train_file', type=str, required=True)
    stm_input.add_argument('--valid_file', type=str, required=True)

    stm_output = sub_trainm.add_argument_group("OUTPUT")
    stm_output.add_argument('--model_dir', type=str, required=True)

    stm_train = sub_trainm.add_argument_group("TRAIN MODEL_HYPER")
    stm_train.add_argument(
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
    )
    stm_train.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    stm_train.add_argument(
        "--signal_len",
        type=int,
        default=15,
        required=False,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    # model param
    stm_train.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    stm_train.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    stm_train.add_argument("--class_num", type=int, default=2, required=False)
    stm_train.add_argument("--dropout_rate", type=float, default=0.5, required=False)
    stm_train.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    stm_train.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    stm_train.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    stm_train.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    stm_train.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )
    # BiLSTM model param
    stm_train.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )

    stm_training = sub_trainm.add_argument_group("TRAINING")
    # model training
    stm_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                  "'Ranger' or 'LookaheadAdam', default Adam")
    stm_training.add_argument('--batch_size', type=int, default=512, required=False)
    stm_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                             choices=["StepLR", "ReduceLROnPlateau"],
                             help="StepLR or ReduceLROnPlateau, default StepLR")
    stm_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001. [lr should be lr*world_size when using multi gpus? "
                                  "or lower batch_size?]")
    stm_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    stm_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    stm_training.add_argument('--lr_patience', type=int, default=0, required=False,
                             help="effective in ReduceLROnPlateau. default 0")
    stm_training.add_argument("--max_epoch_num", action="store", default=20, type=int,
                             required=False, help="max epoch num, default 20")
    stm_training.add_argument("--min_epoch_num", action="store", default=5, type=int,
                             required=False, help="min epoch num, default 5")
    stm_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    stm_training.add_argument('--step_interval', type=int, default=500, required=False)
    stm_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")

    stm_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    stm_training.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')
    stm_training.add_argument('--use_compile', type=str, default="no", required=False,
                             help="[EXPERIMENTAL] if using torch.compile, yes or no, "
                                  "default no ('yes' only works in pytorch>=2.0)")
    stm_training.add_argument('--lambda_corr','--a', type=float, default=0.1)

    stm_trainingp = sub_trainm.add_argument_group("TRAINING PARALLEL")
    stm_trainingp.add_argument("--nodes", default=1, type=int,
                              help="number of nodes for distributed training, default 1")
    stm_trainingp.add_argument("--ngpus_per_node", default=2, type=int,
                              help="number of GPUs per node for distributed training, default 2")
    stm_trainingp.add_argument("--dist-url", default="tcp://127.0.0.1:12315", type=str,
                              help="url used to set up distributed training")
    stm_trainingp.add_argument("--node_rank", default=0, type=int,
                              help="node rank for distributed training, default 0")
    stm_trainingp.add_argument("--epoch_sync", action="store_true", default=False,
                              help="if sync model params of gpu0 to other local gpus after per epoch")
    
    sub_trainm.set_defaults(func=main_trainm)

    # common args =====================================================================================
    parser.add_argument(
        "--transfer",
        action="store_true",
        default=False,
        help="weather use transfer learning",
    )
    parser.add_argument(
        "--domain",
        action="store_true",
        default=False,
        help="weather use domain attribute",
    )
    parser.add_argument(
        "--freq", action="store_true", default=False, help="weather use freq attribute"
    )
    # parser.add_argument(
    #     "--pod5", action="store_true", default=False, help="input pod5 format"
    # )

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main())
