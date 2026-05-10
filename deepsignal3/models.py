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

from einops import repeat, reduce, rearrange
from .utils.sampling import DownsampleLayer
from .utils.utils_mtm import *

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

        self.nhid_seq = self.hidden_size // 2
        self.nhid_signal = self.hidden_size - self.nhid_seq

        # seq feature
        self.embed = nn.Embedding(vocab_size, embedding_size)  # for dna/rna base
        self.is_base = is_base
        self.is_signallen = is_signallen
        self.is_trace = is_trace
        # forward always uses kmer_embed + base_means + base_stds + base_signal_lens (3 scalar feats)
        self.sigfea_num = 3

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
        self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
        # self.dropout_seq = nn.Dropout(p=dropout_rate)
        self.relu_seq = nn.ReLU()

        # signal feature
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
        #self.save_hyperparameters()
        
    # def training_step(self, batch, batch_idx):
    #     kmer, base_means, base_stds, base_signal_lens, signals, labels = batch
    #     outputs, _ = self(kmer, base_means, base_stds, base_signal_lens, signals)
    #     loss = F.cross_entropy(outputs, labels)
    #     self.log('train_loss', loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     kmer, base_means, base_stds, base_signal_lens, signals, labels = batch
    #     outputs, _ = self(kmer, base_means, base_stds, base_signal_lens, signals)
    #     loss = F.cross_entropy(outputs, labels)
    #     self.log('val_loss', loss)
    #     return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
            


        out_seq, _ = self.lstm_seq(
            out_seq,
            self.init_hidden(out_seq.size(0), self.num_layers2, self.nhid_seq, out_seq.device),
        )  # (N, L, nhid_seq*2)
        out_seq = self.fc_seq(out_seq)  # (N, L, nhid_seq)
        # out_seq = self.dropout_seq(out_seq)
        out_seq = self.relu_seq(out_seq)

        # signal feature ==========================================
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


class TokenMixingLayer(nn.Module):

    def __init__(self, d_model=64, r_hid=4, drop=0.2, norm_first=False, temporal_depth=2):
        super().__init__()
                
        self.temporal = nn.ModuleList([
            TemporalAttn(d_model, drop, norm_first)
            for _ in range(temporal_depth)
        ])
        
        self.mlp3 = MLP(d_model, r_hid, drop, norm_first)
        


    def forward(self, x, x_mask, cls_tok, pos, pe, idx_c, pe_type='rel'):
        x = torch.concat([cls_tok, x], dim=1)
        x_mask = F.pad(x_mask, (0, 0, 1, 0), 'constant', False)
        p_mask = pos < 0
        pos = F.pad(pos + 1, (0, 0, 1, 0), 'constant', 0)
        p_mask = F.pad(p_mask, (0, 0, 1, 0), 'constant', False)
        
        
        imp = None
        for layer in self.temporal:
            x, imp = layer(x, x_mask, pos, pe, pe_type)                  
                
        x = self.mlp3(x)
        return x[:, 1:, :, :], x[:, [0], :, :], imp


class modelMTM(nn.Module):

    def __init__(self, num_chn, d_static, num_cls, ratios, d_model=96, r_hid=4, drop=0.2, norm_first=True, down_mode='concat', 
                 vocab_size=16, embedding_size=4, temporal_depth=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_static = d_static
        self.ratios = ratios
        self.register_buffer('rpe', precompute_rpe(d_model))
        self.register_buffer('ape', precompute_ape(d_model))
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.chn_emb = nn.Embedding(num_chn, d_model)
        nn.init.xavier_uniform_(self.chn_emb.weight)
        self.cls_tok = nn.Parameter(torch.rand(num_chn, d_model))
        self.register_buffer('_c_arange', torch.arange(num_chn), persistent=False)

        # 传入 temporal_depth
        self.inp_layer = TokenMixingLayer(d_model, r_hid, drop, norm_first, temporal_depth)
        self.mixers = nn.ModuleList()
        self.samplers = nn.ModuleList()
        for r in ratios:
            self.mixers.append(TokenMixingLayer(d_model, r_hid, drop, norm_first, temporal_depth))
            self.samplers.append(DownsampleLayer(d_model, r, down_mode))

        self.cls_head = CLSHead(d_model, d_static, num_cls, drop)

  
    def forward(self, signal, kmer, x_mask, t, x_static):
        kmer_embed = self.embedding(kmer)
        x = torch.cat([signal, kmer_embed], dim=-1)

        bsz, nt, nc = x_mask.shape
        nt = nt + 1
        dev = x.device
        idx_t = repeat(t, "b t -> b t c", c=nc)
        idx_b = repeat(torch.arange(bsz, device=dev), "b -> b t c", t=nt, c=nc)
        idx_c = repeat(self._c_arange, "c -> b t c", b=bsz, t=nt)
        c_feat = self.chn_emb(self._c_arange)

        x = apply_abs_pe(x.nan_to_num(0)[..., None] * c_feat, idx_t, self.ape)
        cls_tok = repeat(self.cls_tok, "c d -> b 1 c d", b=bsz)

        x, cls_tok, imp = self.inp_layer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        for sampler, mixer in zip(self.samplers, self.mixers):
            x, x_mask, idx_t = sampler(x, x_mask, idx_b, idx_t, idx_c, imp)
            x, cls_tok, imp = mixer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        outputs = [reduce(cls_tok, "b 1 c d -> b d", 'max')]
        if self.d_static > 0:
            outputs.append(x_static)
        outputs = self.cls_head(torch.cat(outputs, -1))
        return outputs


########################################
class AggrAttRNN(nn.Module):
    def __init__(self, seq_len=11, num_layers=1, num_classes=1,
                 dropout_rate=0.5, hidden_size=32, binsize=20,
                 model_type="attbigru",
                 device=0):
        super(AggrAttRNN, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.feas_ccs = binsize + 1
        if self.model_type == "attbilstm":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=0, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=0, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.num_classes)  # 2 for bidirection

        # self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
        if use_cuda and self.device != "cpu":
            h0 = h0.cuda(self.device)
        if self.rnn_cell == "lstm":
            c0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
            if use_cuda and self.device != "cpu":
                c0 = c0.cuda(self.device)
            return h0, c0
        return h0

    def forward(self, offsets, histos):

        offsets = torch.reshape(offsets, (-1, self.seq_len, 1)).float()  # (N, L, 1)

        out = torch.cat((histos.float(), offsets), 2)

        out, n_states = self.rnn(out, self.init_hidden(out.size(0),
                                                       self.num_layers,
                                                       self.hidden_size))  # (N, L, nhid*2)
        # attention_net3 ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        # h_n (last layer) = out[:, -1, :self.hidden_size] concats out1[:, 0, self.hidden_size:]
        h_n = n_states[0] if self.rnn_cell == "lstm" else n_states
        h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n = h_n.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out, att_weights = self._att3(h_n, out)

        out = self.dropout1(out)
        out = self.fc1(out)
        # out = self.softmax(out)

        return out


class ReadCalibRNN(nn.Module):
    """Read-level calibration: uses K neighboring CpG sites on the same read
    to calibrate the methylation probability of a target site.

    Symmetric to AggrAttRNN — where AggrAttRNN aggregates N reads at one site,
    ReadCalibRNN aggregates K co-read sites for one target.

    Input (per batch):
        window_probs:   (N, K)     HTE prob_1 of K window positions
        window_offsets: (N, K)     genomic distance from target (bp)
        window_valid:   (N, K)     1.0 = real CpG, 0.0 = padding
        read_feats:     (N, n_rf)  read-level features [log1p(n_cpg), mean_prob, std_prob]
    Output:
        logit: (N, 1) — apply sigmoid to get calibrated probability
    """

    def __init__(self, seq_len=11, num_layers=1, num_classes=1,
                 dropout_rate=0.5, hidden_size=32, n_read_feats=3,
                 model_type='attbigru'):
        super(ReadCalibRNN, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.n_read_feats = n_read_feats

        # 3 features per window position: prob, offset_norm, is_valid
        self.input_dim = 3

        if model_type == 'attbigru':
            self.rnn_cell = 'gru'
            self.rnn = nn.GRU(self.input_dim, hidden_size, num_layers,
                              dropout=0, batch_first=True, bidirectional=True)
        elif model_type == 'attbilstm':
            self.rnn_cell = 'lstm'
            self.rnn = nn.LSTM(self.input_dim, hidden_size, num_layers,
                               dropout=0, batch_first=True, bidirectional=True)
        else:
            raise ValueError("model_type must be 'attbigru' or 'attbilstm'")

        self._att = Attention(hidden_size * 2, hidden_size * 2, hidden_size)

        # project read-level features to match context size
        self.read_feat_proj = nn.Sequential(
            nn.Linear(n_read_feats, hidden_size * 2),
            nn.ReLU(),
        )

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def get_model_type(self):
        return self.model_type

    def forward(self, window_probs, window_offsets, window_valid, read_feats):
        # window_probs, window_offsets, window_valid: (N, K)
        # read_feats: (N, n_rf)
        offsets_norm = (window_offsets.float() / 1000.0).clamp(0.0, 10.0)
        x = torch.stack([window_probs.float(), offsets_norm, window_valid.float()], dim=-1)  # (N, K, 3)

        out, n_states = self.rnn(x)  # out: (N, K, 2H)

        h_n = n_states[0] if self.rnn_cell == 'lstm' else n_states
        h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer: (2, N, H)
        h_n = h_n.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)   # (N, 1, 2H)

        context, _ = self._att(h_n, out)  # (N, 2H)

        rf = self.read_feat_proj(read_feats.float())  # (N, 2H)
        combined = torch.cat([context, rf], dim=-1)   # (N, 4H)

        out = self.dropout1(combined)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)   # (N, 1)
        return out


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


# bahdanau attention
class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)  # query, (2, N, C) -> (N, 1, C*2)
        encoder_outputs: (batch_size, max_time, hidden_size)  # key, (N, L, C*2)
    Returns:
        context_vector: (N, 2*C)
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, query_size, key_size, hidden_size=128):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.Wa = nn.Linear(query_size, hidden_size, bias=False)
        self.Ua = nn.Linear(key_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=False)
        self.attw_softmax = nn.Softmax(1)

    def forward(self, last_hidden, encoder_outputs):

        attention_energies = self._score(last_hidden, encoder_outputs).squeeze(2)  # (N, L, 1) -> (N, L)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        attention_weights = self.attw_softmax(attention_energies).unsqueeze(2)  # (N, L) -> (N, L, 1)

        values = torch.transpose(encoder_outputs, 1, 2)  # (N, 2*C, L)
        context_vector = torch.matmul(values, attention_weights).squeeze(2)  # (N, 2*C, 1) -> (N, 2*C)

        return context_vector, attention_weights.squeeze(2)

    def _score(self, last_hidden, encoder_outputs):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)  # (2, N, C) -> (N, 1, C*2)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)  # (N, L, C*2)
        :return: a score (batch_size, max_time)
        """
        out = torch.tanh(self.Wa(last_hidden) + self.Ua(encoder_outputs))  # (N, L, nhid)
        return self.va(out)  # (N, L, 1)