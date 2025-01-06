#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
#import pytorch_lightning as pl
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torch.utils
import torch.utils.checkpoint

from .utils.constants_torch import use_cuda
import numpy as np

# inner module ================================================
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""


class BasicBlock(nn.Module):
    """use Conv1d and BatchNorm1d"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_3layers(nn.Module):
    """Conv1d"""

    def __init__(
        self, block, num_blocks, strides, out_channels=128, init_channels=1, in_planes=4
    ):
        super(ResNet_3layers, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv1d(
            init_channels,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        # three group of blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(
            block, out_channels, num_blocks[2], stride=strides[2]
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # (N, 1, L) --> (N, 4, L)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


def get_lout(lin, strides):
    import math

    lout = lin
    for stride in strides:
        lout = math.floor(float(lout - 1) / stride + 1)
    return lout


def ResNet3(out_channels=128, strides=(1, 2, 2), init_channels=1, in_planes=4):
    """ResNet with 3 blocks"""
    return ResNet_3layers(
        BasicBlock, [1, 1, 1], strides, out_channels, init_channels, in_planes
    )


# model ===============================================
class ModelBiLSTM(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        #device=0,
    ):
        super(ModelBiLSTM, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        #self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 3 if self.is_signallen else 2
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    embedding_size + self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
                self.lstm_seq.flatten_parameters()
                # (batch_size,seq_len,hidden_size*2)
            else:
                self.lstm_seq = nn.LSTM(
                    self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
                self.lstm_seq.flatten_parameters()
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(
                self.signal_len,
                self.nhid_signal,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            self.lstm_signal.flatten_parameters()
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_comb.flatten_parameters()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size, device):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size)).to(device)
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size)).to(device)
        # if use_cuda:
        #     h0 = h0.cuda(self.device)
        #     c0 = c0.cuda(self.device)
        return h0, c0

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals):
        # seq feature ============================================
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(
                base_signal_lens, (-1, self.seq_len, 1)
            ).float()
            # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                # print("base_means: ")
                # print(base_means.shape)
                # print("base_stds: ")
                # print(base_stds.shape)
                # print("base_signal_lens: ")
                # print(base_signal_lens.shape)
                # print("kmer_embed: ")
                # print(kmer_embed.shape)
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
            else:
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)

            out_seq, _ = self.lstm_seq(
                out_seq,
                self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq, out_seq.device),
            )  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # print("signals: ")
            # print(signals.shape)
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(
                out_signal,
                self.init_hidden(
                    out_signal.size(0), self.num_layers2, self.nhid_signal, out_signal.device
                ),
            )
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size,out.device)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


class ModelExtraction(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
        lambd=1.0,
    ):
        super(ModelExtraction, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.lambd = lambd

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 3 if self.is_signallen else 2
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    embedding_size + self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            else:
                self.lstm_seq = nn.LSTM(
                    self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(
                self.signal_len,
                self.nhid_signal,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    # def backward(self, grad_output):
    #    # 在反向传播中，将梯度乘以-lambd
    #    return -self.lambd * grad_output

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals):
        # seq feature ============================================
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(
                base_signal_lens, (-1, self.seq_len, 1)
            ).float()
            # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
            else:
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)

            out_seq, _ = self.lstm_seq(
                out_seq,
                self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq),
            )  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(
                out_signal,
                self.init_hidden(
                    out_signal.size(0), self.num_layers2, self.nhid_signal
                ),
            )
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        combine_out = torch.cat((out_fwd_last, out_bwd_last), 1)

        return combine_out


class ModelDomainExtraction(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
        lambd=1.0,
    ):
        super(ModelDomainExtraction, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.lambd = lambd

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 4 if self.is_signallen else 3
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    embedding_size + self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            else:
                self.lstm_seq = nn.LSTM(
                    self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(
                self.signal_len,
                self.nhid_signal,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    # def backward(self, grad_output):
    #    # 在反向传播中，将梯度乘以-lambd
    #    return -self.lambd * grad_output

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals, tags):
        # seq feature ============================================
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(
                base_signal_lens, (-1, self.seq_len, 1)
            ).float()
            # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
            tags = torch.reshape(tags, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens, tags), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens, tags), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, tags), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, tags), 2
                    )  # (N, L, C)
            else:
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens, tags), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens, tags), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat((base_means, base_stds, tags), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((base_means, base_stds, tags), 2)  # (N, L, C)

            out_seq, _ = self.lstm_seq(
                out_seq,
                self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq),
            )  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(
                out_signal,
                self.init_hidden(
                    out_signal.size(0), self.num_layers2, self.nhid_signal
                ),
            )
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        combine_out = torch.cat((out_fwd_last, out_bwd_last), 1)

        return combine_out


class combineLoss(nn.Module):
    def __init__(self, device=0, β=0.1):
        super(combineLoss, self).__init__()
        # weight_rank = torch.from_numpy(np.array([1, 1.0])).float()
        weight_rank = torch.from_numpy(np.array([1,1, 1.0])).float()
        self.device = device
        self.β = β
        if use_cuda:
            weight_rank = weight_rank.cuda(self.device)
            # self.β=β.cuda(self.device)
            # weight_rank2 = weight_rank2.cuda(self.device)
        self.loss = nn.CrossEntropyLoss(weight=weight_rank)#nn.BCEWithLogitsLoss(pos_weight=weight_rank)  # nn.BCELoss()
        self.project = nn.Sigmoid()
        # self.loss_2 = nn.CrossEntropyLoss(weight=weight_rank2)

    def forward(self, domain_classes, tags):
        # classes = classes.reshape(classes.shape[0], 2)
        # labels = labels.reshape(labels.shape[0], 1)
        # print('classes shape: {}'.format(classes.shape))
        # print('labels shape: {}'.format(labels.shape))
        # left = F.relu(0.9 - classes[0], inplace=True) ** 2
        # print('left shape: {}'.format(left.shape))
        # right = F.relu(classes[1] - 0.1, inplace=True) ** 2
        # print('right shape: {}'.format(right.shape))

        # margin_loss = labels * left + 0.5 * (1.0 - labels) * right
        # margin_loss = margin_loss.sum()
        onehot_tags = torch.eye(3)[tags.long(), :].cuda(self.device)
        return self.β * self.loss(domain_classes, onehot_tags)


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coeff=1.0):
        ctx.coeff = coeff
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self, coeff=1.0):
        super(GradientReverseLayer, self).__init__()
        self.coeff = coeff

    def forward(self, input):
        return GradientReverseFunction.apply(input, self.coeff)


class Classifier1(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_size=256, num_classes=2, device=0):
        super(Classifier1, self).__init__()
        self.device = device

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, combine_out):
        # decode
        out = self.dropout(combine_out)  # .cuda(self.device)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc1(out)
        return out, self.softmax(out)


class Classifier2(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_size=256, num_classes=3, device=0):
        super(Classifier2, self).__init__()
        self.device = device
        self.grl = GradientReverseLayer()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(1)

    def forward(self, combine_out):
        # decode
        out = self.grl(combine_out)
        out = self.dropout(out)  # .cuda(self.device)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc1(out)
        return out


class ModelCNN(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
    ):
        super(ModelCNN, self).__init__()
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        self.nhid_seq = self.hidden_size // 2
        self.nhid_signal = self.hidden_size - self.nhid_seq

        # seq feature
        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.is_trace = is_trace
        self.sigfea_num = 3 if self.is_signallen else 2
        # (batch_size,seq_len,embedding_size+sigfea_num)
        cnn_seq_out = int((2 * self.nhid_seq * self.hidden_size) / 4)
        self.cnn_seq = nn.Conv1d(self.seq_len, cnn_seq_out, 4)
        self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
        # self.dropout_seq = nn.Dropout(p=dropout_rate)
        self.relu_seq = nn.ReLU()

        # signal feature

        # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
        cnn_signal_out = int((2 * self.nhid_signal * self.hidden_size) / 8)
        self.cnn_signal = nn.Conv1d(self.seq_len, cnn_signal_out, 9)
        self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal = nn.ReLU()

        # combined
        self.cnn_comb = nn.Conv1d(self.hidden_size, self.hidden_size * 2, 1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, kmer, base_means, base_stds, base_signal_lens, signals):
        # seq feature ============================================
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(
            base_signal_lens, (-1, self.seq_len, 1)
        ).float()
        # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
        kmer_embed = self.embed(kmer.long())
        out_seq = torch.cat(
            (kmer_embed, base_means, base_stds, base_signal_lens), 2
        )  # (N, L, C)

        out_seq = self.cnn_seq(out_seq)  # (N, L, nhid_seq*2)
        out_seq = torch.reshape(
            out_seq, (-1, self.nhid_seq * 2, self.hidden_size)
        ).float()
        out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
        # out_seq = self.dropout_seq(out_seq)
        out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        out_signal = signals.float()
        # resnet ---
        # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
        # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
        # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
        # lstm ---
        out_signal = self.cnn_signal(out_signal)
        out_signal = torch.reshape(
            out_signal, (-1, self.nhid_signal * 2, self.hidden_size)
        ).float()
        out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
        # out_signal = self.dropout_signal(out_signal)
        out_signal = self.relu_signal(out_signal)

        # combined ================================================
        out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out = self.cnn_comb(out)  # (N, L, hidden_size*2)
        out = torch.reshape(out, (-1, self.hidden_size, self.hidden_size * 2)).float()
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


class ModelCG(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
    ):
        super(ModelCG, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 5 if self.is_signallen else 4
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    embedding_size + self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
                # (batch_size,seq_len,hidden_size*2)
            else:
                self.lstm_seq = nn.LSTM(
                    self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(
                self.signal_len,
                self.nhid_signal,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(
        self, kmer, base_means, base_stds, base_signal_lens, signals, tags, cg_contents
    ):
        # seq feature ============================================
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(
                base_signal_lens, (-1, self.seq_len, 1)
            ).float()
            tags = torch.reshape(tags, (-1, self.seq_len, 1)).float()
            cg_contents = torch.reshape(cg_contents, (-1, self.seq_len, 1)).float()
            # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (
                            kmer_embed,
                            base_means,
                            base_stds,
                            base_signal_lens,
                            tags,
                            cg_contents,
                        ),
                        2,
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (
                            kmer_embed,
                            base_means,
                            base_stds,
                            base_signal_lens,
                            tags,
                            cg_contents,
                        ),
                        2,
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, tags, cg_contents), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, tags, cg_contents), 2
                    )  # (N, L, C)
            else:
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens, tags, cg_contents), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens, tags, cg_contents), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, tags, cg_contents), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (base_means, base_stds, tags, cg_contents), 2
                    )  # (N, L, C)

            out_seq, _ = self.lstm_seq(
                out_seq,
                self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq),
            )  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(
                out_signal,
                self.init_hidden(
                    out_signal.size(0), self.num_layers2, self.nhid_signal
                ),
            )
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


class ModelCombine(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
    ):
        super(ModelCombine, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        if self.module != "signal_bilstm":
            self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
            self.is_base = is_base
            self.is_signallen = is_signallen
            self.is_trace = is_trace
            self.sigfea_num = 3 if self.is_signallen else 2
            if self.is_trace:
                self.sigfea_num += 1
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    embedding_size + self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
                # (batch_size,seq_len,hidden_size*2)
            else:
                self.lstm_seq = nn.LSTM(
                    self.sigfea_num,
                    self.nhid_seq,
                    self.num_layers2,
                    dropout=dropout_rate,
                    batch_first=True,
                    bidirectional=True,
                )
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            # self.dropout_seq = nn.Dropout(p=dropout_rate)
            self.relu_seq = nn.ReLU()

        # signal feature
        if self.module != "seq_bilstm":
            # self.convs = ResNet3(self.nhid_signal, (1, 1, 1), self.signal_len, self.signal_len)  # (N, C, L)
            self.lstm_signal = nn.LSTM(
                self.signal_len,
                self.nhid_signal,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            # self.dropout_signal = nn.Dropout(p=dropout_rate)
            self.relu_signal = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(
        self, kmer, base_means, base_stds, base_signal_lens, signals, tags, cg_contents
    ):
        # seq feature ============================================
        if self.module != "signal_bilstm":
            base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
            base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
            base_signal_lens = torch.reshape(
                base_signal_lens, (-1, self.seq_len, 1)
            ).float()
            tags = tags.float()
            cg_contents = cg_contents.float()

            # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
            if self.is_base:
                kmer_embed = self.embed(kmer.long())
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
                else:
                    out_seq = torch.cat(
                        (kmer_embed, base_means, base_stds), 2
                    )  # (N, L, C)
            else:
                if self.is_signallen and self.is_trace:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_signallen:
                    out_seq = torch.cat(
                        (base_means, base_stds, base_signal_lens), 2
                    )  # (N, L, C)
                elif self.is_trace:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
                else:
                    out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)

            out_seq, _ = self.lstm_seq(
                out_seq,
                self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq),
            )  # (N, L, nhid_seq*2)
            out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
            # out_seq = self.dropout_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        if self.module != "seq_bilstm":
            out_signal = signals.float()
            # resnet ---
            # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
            # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
            # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
            # lstm ---
            out_signal, _ = self.lstm_signal(
                out_signal,
                self.init_hidden(
                    out_signal.size(0), self.num_layers2, self.nhid_signal
                ),
            )
            out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
            # out_signal = self.dropout_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        # combined ================================================
        if self.module == "seq_bilstm":
            out = out_seq
        elif self.module == "signal_bilstm":
            out = out_signal
        elif self.module == "both_bilstm":
            out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)
        extrac_fea = torch.cat((tags, cg_contents), 1)
        out = torch.cat((out, extrac_fea), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


class ModelFrequency(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
    ):
        super(ModelFrequency, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.is_trace = is_trace
        self.sigfea_num = 3 if self.is_signallen else 2
        if self.is_trace:
            self.sigfea_num += 1
        if self.is_base:
            self.lstm_seq = nn.LSTM(
                embedding_size + self.sigfea_num,
                self.nhid_seq,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            # (batch_size,seq_len,hidden_size*2)
        else:
            self.lstm_seq = nn.LSTM(
                self.sigfea_num,
                self.nhid_seq,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
        self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
        # self.dropout_seq = nn.Dropout(p=dropout_rate)
        self.relu_seq = nn.ReLU()

        # signal feature
        self.lstm_signal = nn.LSTM(
            self.signal_len,
            self.nhid_signal,
            self.num_layers2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal = nn.ReLU()

        # signal frequency feature
        self.lstm_signal_freq = nn.LSTM(
            self.signal_len,
            self.nhid_signal,
            self.num_layers2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_signal_freq = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal_freq = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size + self.nhid_signal,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(
        self, kmer, base_means, base_stds, base_signal_lens, signals, signals_freq
    ):
        # seq feature ============================================
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(
            base_signal_lens, (-1, self.seq_len, 1)
        ).float()
        # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
        if self.is_base:
            kmer_embed = self.embed(kmer.long())
            if self.is_signallen and self.is_trace:
                out_seq = torch.cat(
                    (kmer_embed, base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_signallen:
                out_seq = torch.cat(
                    (kmer_embed, base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_trace:
                out_seq = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
            else:
                out_seq = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
        else:
            if self.is_signallen and self.is_trace:
                out_seq = torch.cat(
                    (base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_signallen:
                out_seq = torch.cat(
                    (base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_trace:
                out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
            else:
                out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)

        out_seq, _ = self.lstm_seq(
            out_seq, self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq)
        )  # (N, L, nhid_seq*2)
        out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
        # out_seq = self.dropout_seq(out_seq)
        out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        out_signal = signals.float()
        # resnet ---
        # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
        # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
        # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
        # lstm ---
        out_signal, _ = self.lstm_signal(
            out_signal,
            self.init_hidden(out_signal.size(0), self.num_layers2, self.nhid_signal),
        )
        out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
        # out_signal = self.dropout_signal(out_signal)
        out_signal = self.relu_signal(out_signal)

        # signal feature ==========================================
        out_signal_freq = signals_freq.float()
        out_signal_freq, _ = self.lstm_signal_freq(
            out_signal_freq,
            self.init_hidden(
                out_signal_freq.size(0), self.num_layers2, self.nhid_signal
            ),
        )
        out_signal_freq = self.fc_signal_freq(out_signal_freq)
        out_signal_freq = self.relu_signal(out_signal_freq)

        # combined ================================================
        out = torch.cat(
            (out_seq, out_signal, out_signal_freq), 2
        )  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)


class ModelFrequency_mp(nn.Module):
    def __init__(
        self,
        seq_len=13,
        signal_len=16,
        num_layers1=3,
        num_layers2=1,
        num_classes=2,
        dropout_rate=0.5,
        hidden_size=256,
        vocab_size=16,
        embedding_size=4,
        is_base=True,
        is_signallen=True,
        is_trace=False,
        module="both_bilstm",
        device=0,
    ):
        super(ModelFrequency_mp, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1  # for combined (seq+signal) feature
        self.num_layers2 = num_layers2  # for seq and signal feature separately
        self.num_classes = num_classes

        self.hidden_size = hidden_size

        if self.module == "both_bilstm":
            self.nhid_seq = self.hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "seq_bilstm":
            self.nhid_seq = self.hidden_size
        elif self.module == "signal_bilstm":
            self.nhid_signal = self.hidden_size
        else:
            raise ValueError("--model_type is not right!")

        # seq feature
        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.is_trace = is_trace
        self.sigfea_num = 3 if self.is_signallen else 2
        if self.is_trace:
            self.sigfea_num += 1
        if self.is_base:
            self.lstm_seq = nn.LSTM(
                embedding_size + self.sigfea_num,
                self.nhid_seq,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
            # (batch_size,seq_len,hidden_size*2)
        else:
            self.lstm_seq = nn.LSTM(
                self.sigfea_num,
                self.nhid_seq,
                self.num_layers2,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True,
            )
        self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
        # self.dropout_seq = nn.Dropout(p=dropout_rate)
        self.relu_seq = nn.ReLU()

        # signal feature
        self.lstm_signal = nn.LSTM(
            self.signal_len,
            self.nhid_signal,
            self.num_layers2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal = nn.ReLU()

        # signal frequency Phase feature
        self.lstm_signal_freq_p = nn.LSTM(
            self.signal_len,
            self.nhid_signal,
            self.num_layers2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_signal_freq_p = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal_freq_p = nn.ReLU()

        # signal frequency Magnitude feature
        self.lstm_signal_freq_m = nn.LSTM(
            self.signal_len,
            self.nhid_signal,
            self.num_layers2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_signal_freq_m = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
        # self.dropout_signal = nn.Dropout(p=dropout_rate)
        self.relu_signal_freq_m = nn.ReLU()

        # combined
        self.lstm_comb = nn.LSTM(
            self.hidden_size * 2,
            self.hidden_size,
            self.num_layers1,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # 2 for bidirection
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = autograd.Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if use_cuda:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(
        self, kmer, base_means, base_stds, base_signal_lens, signals, magnitude, phase
    ):
        # seq feature ============================================
        base_means = torch.reshape(base_means, (-1, self.seq_len, 1)).float()
        base_stds = torch.reshape(base_stds, (-1, self.seq_len, 1)).float()
        base_signal_lens = torch.reshape(
            base_signal_lens, (-1, self.seq_len, 1)
        ).float()
        # base_probs = torch.reshape(base_probs, (-1, self.seq_len, 1)).float()
        if self.is_base:
            kmer_embed = self.embed(kmer.long())
            if self.is_signallen and self.is_trace:
                out_seq = torch.cat(
                    (kmer_embed, base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_signallen:
                out_seq = torch.cat(
                    (kmer_embed, base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_trace:
                out_seq = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
            else:
                out_seq = torch.cat((kmer_embed, base_means, base_stds), 2)  # (N, L, C)
        else:
            if self.is_signallen and self.is_trace:
                out_seq = torch.cat(
                    (base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_signallen:
                out_seq = torch.cat(
                    (base_means, base_stds, base_signal_lens), 2
                )  # (N, L, C)
            elif self.is_trace:
                out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)
            else:
                out_seq = torch.cat((base_means, base_stds), 2)  # (N, L, C)

        out_seq, _ = self.lstm_seq(
            out_seq, self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq)
        )  # (N, L, nhid_seq*2)
        out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
        # out_seq = self.dropout_seq(out_seq)
        out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
        out_signal = signals.float()
        # resnet ---
        # out_signal = out_signal.transpose(1, 2)  # (N, C, L)
        # out_signal = self.convs(out_signal)  # (N, nhid_signal, L)
        # out_signal = out_signal.transpose(1, 2)  # (N, L, nhid_signal)
        # lstm ---
        out_signal, _ = self.lstm_signal(
            out_signal,
            self.init_hidden(out_signal.size(0), self.num_layers2, self.nhid_signal),
        )
        out_signal = self.fc_signal(out_signal)  # (N, L, nhid_signal)
        # out_signal = self.dropout_signal(out_signal)
        out_signal = self.relu_signal(out_signal)

        # signal magnitude feature ==========================================
        out_signal_freq_m = magnitude.float()
        out_signal_freq_m, _ = self.lstm_signal_freq_m(
            out_signal_freq_m,
            self.init_hidden(
                out_signal_freq_m.size(0), self.num_layers2, self.nhid_signal
            ),
        )
        out_signal_freq_m = self.fc_signal_freq_m(out_signal_freq_m)
        out_signal_freq_m = self.relu_signal_freq_m(out_signal_freq_m)

        # signal magnitude feature ==========================================
        out_signal_freq_p = phase.float()
        out_signal_freq_p, _ = self.lstm_signal_freq_p(
            out_signal_freq_p,
            self.init_hidden(
                out_signal_freq_p.size(0), self.num_layers2, self.nhid_signal
            ),
        )
        out_signal_freq_p = self.fc_signal_freq_p(out_signal_freq_p)
        out_signal_freq_p = self.relu_signal_freq_p(out_signal_freq_p)

        # combined ================================================
        out = torch.cat(
            (out_seq, out_signal, out_signal_freq_m, out_signal_freq_p), 2
        )  # (N, L, hidden_size)
        out, _ = self.lstm_comb(
            out, self.init_hidden(out.size(0), self.num_layers1, self.hidden_size)
        )  # (N, L, hidden_size*2)
        out_fwd_last = out[:, -1, : self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size :]
        out = torch.cat((out_fwd_last, out_bwd_last), 1)

        # decode
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out, self.softmax(out)
