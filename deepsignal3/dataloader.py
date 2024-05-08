from torch.utils.data import Dataset
import linecache
import os
import numpy as np

from .utils.process_utils import base2code_dna
import pywt


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


def parse_a_line1(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    # tag = int(words[13])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
    )  # , tag


def parse_a_line2(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    tag = int(words[12])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        tag,
    )


def parse_a_line3(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    tag = np.array([int(words[12])] * base_signal_lens.shape[0])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        tag,
    )


def parse_a_line4(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    tag = np.array([int(words[12])] * base_signal_lens.shape[0])
    cg_content = np.array([float(words[13])] * base_signal_lens.shape[0])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        tag,
        cg_content,
    )


def parse_a_line5(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    tag = np.array([int(words[12])])
    cg_content = np.array([float(words[13])])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        tag,
        cg_content,
    )


def parse_a_line6(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    signals_freq = np.reshape(
        np.fft.fft(k_signals.flatten()),
        (np.shape(k_signals)[0], np.shape(k_signals)[1]),
    )  # np.array([[float(y) for y in x.split(",")] for x in words[13].split(";")])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        np.abs(signals_freq),
        np.angle(signals_freq),
    )


def parse_a_line7(line):
    words = line.strip().split()

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    # base_probs = np.zeros(base_signal_lens.shape[0])
    # base_probs = np.array([float(x) for x in words[10].split(",")])

    k_signals = np.array(
        [[float(y) for y in x.split(",")] for x in words[10].split(";")]
    )
    label = int(words[11])
    # 进行小波变换
    wavelet = "db4"  # 小波基函数，这里选择 Daubechies 4
    level = 5  # 分解级数
    coeffs = pywt.wavedec(k_signals.flatten(), wavelet, level=level)
    # 设计高通滤波器
    high_pass_filter = np.array([1, -1])  # 一阶差分滤波器

    # 将高通滤波器应用于小波系数
    filtered_coeffs = [np.convolve(c, high_pass_filter, mode="same") for c in coeffs]
    signals_freq = np.reshape(
        pywt.waverec(filtered_coeffs, wavelet),
        (np.shape(k_signals)[0], np.shape(k_signals)[1]),
    )  # reconstructed signal
    # signals_freq=np.reshape(np.fft.fft(k_signals.flatten()),(np.shape(k_signals)[0],np.shape(k_signals)[1]))#np.array([[float(y) for y in x.split(",")] for x in words[13].split(";")])

    return (
        sampleinfo,
        kmer,
        base_means,
        base_stds,
        base_signal_lens,
        k_signals,
        label,
        signals_freq,
    )


class SignalFeaData1(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line1(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData2(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line2(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData3(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line3(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData4(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line4(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData5(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line5(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData6(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line6(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


class SignalFeaData7(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line7(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data
