# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import numpy as np
import argparse
import os
import sys
import time
import re

from .models import (
    ModelExtraction,
    combineLoss,
    Classifier1,
    Classifier2,
    ModelBiLSTM,
    ModelDomainExtraction,
    ModelCNN,
    ModelCG,
    ModelCombine,
    ModelFrequency,
    ModelFrequency_mp,
)
from .dataloader import (
    SignalFeaData1,
    SignalFeaData2,
    SignalFeaData3,
    SignalFeaData4,
    SignalFeaData5,
    SignalFeaData6,
    SignalFeaData7,
)
from .dataloader import clear_linecache
from .utils.process_utils import display_args
from .utils.process_utils import str2bool

from .utils.constants_torch import use_cuda
from .utils import infonce


def train(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData1(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData1(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelBiLSTM(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels = (
                sfeatures
            )
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()

            # Forward pass
            outputs, logits = model(
                kmer, base_means, base_stds, base_signal_lens, signals
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                        voutputs, vlogits = model(
                            vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_transfer(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData2(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData2(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelExtraction(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    classifier1 = Classifier1()
    classifier2 = Classifier2()
    if use_cuda:
        model = model.cuda()
        classifier1 = classifier1.cuda()
        classifier2 = classifier2.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion1 = nn.CrossEntropyLoss(weight=weight_rank)
    criterion2 = combineLoss()
    if args.optim_type == "Adam":
        optimizer1 = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )
        optimizer2 = torch.optim.Adam(
            [
                {"params": classifier2.parameters()},
            ],
            lr=args.lr,
        )
    elif args.optim_type == "RMSprop":
        optimizer1 = torch.optim.RMSprop(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )
        optimizer2 = torch.optim.RMSprop(
            [
                {"params": classifier2.parameters()},
            ],
            lr=args.lr,
        )
    elif args.optim_type == "SGD":
        optimizer1 = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
            momentum=0.8,
        )
        optimizer2 = torch.optim.SGD(
            [
                {"params": classifier2.parameters()},
            ],
            lr=args.lr,
            momentum=0.8,
        )
    else:
        raise ValueError("optim_type is not right!")
    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.1)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.5)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    classifier1.train()
    classifier2.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels, tags = (
                sfeatures
            )
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                tags = tags.cuda()

            # Forward pass
            combine_outputs = model(
                kmer, base_means, base_stds, base_signal_lens, signals
            )
            outputs, _ = classifier1(combine_outputs)
            domain_out = classifier2(combine_outputs)
            loss1 = criterion1(
                outputs,
                labels,
            )
            loss2 = criterion2(
                domain_out,
                tags,
            )
            loss = loss1
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward()
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier1.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier2.parameters(), 0.5)
            optimizer1.step()
            optimizer2.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                classifier1.eval()
                classifier2.eval()
                with torch.no_grad():
                    vlosses1, vlosses2, vaccus, vprecs, vrecas = [], [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vtags,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vtags = vtags.cuda()
                        vcombine_outputs = model(
                            vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals
                        )
                        voutputs, vlogits = classifier1(vcombine_outputs)
                        vdomain_out = classifier2(vcombine_outputs)
                        vloss1 = criterion1(
                            voutputs,
                            vlabels,
                        )
                        vloss2 = criterion2(
                            vdomain_out,
                            vtags,
                        )

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses1.append(vloss1.item())
                        vlosses2.append(vloss2.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            torch.save(
                                classifier1.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.classifier1.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            torch.save(
                                classifier2.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.classifier2.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss1: {:.4f},ValidLoss2: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses1),
                            np.mean(vlosses2),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
                classifier1.train()
                classifier2.train()
        scheduler1.step()
        scheduler2.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_domain(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData3(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData3(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelDomainExtraction(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    classifier1 = Classifier1()
    if use_cuda:
        model = model.cuda()
        classifier1 = classifier1.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion1 = nn.CrossEntropyLoss(weight=weight_rank)
    criterion2 = combineLoss()
    if args.optim_type == "Adam":
        optimizer1 = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )
    elif args.optim_type == "RMSprop":
        optimizer1 = torch.optim.RMSprop(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )

    elif args.optim_type == "SGD":
        optimizer1 = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
            momentum=0.8,
        )

    else:
        raise ValueError("optim_type is not right!")
    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    classifier1.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels, tags = (
                sfeatures
            )
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                tags = tags.cuda()

            # Forward pass
            combine_outputs = model(
                kmer, base_means, base_stds, base_signal_lens, signals, tags
            )
            outputs, _ = classifier1(combine_outputs)
            loss1 = criterion1(
                outputs,
                labels,
            )
            loss = loss1
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer1.zero_grad()
            loss1.backward(retain_graph=True)
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier1.parameters(), 0.5)
            optimizer1.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                classifier1.eval()
                with torch.no_grad():
                    vlosses1, vlosses2, vaccus, vprecs, vrecas = [], [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vtags,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vtags = vtags.cuda()
                        vcombine_outputs = model(
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vtags,
                        )
                        voutputs, vlogits = classifier1(vcombine_outputs)
                        vloss1 = criterion1(
                            voutputs,
                            vlabels,
                        )

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses1.append(vloss1.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            torch.save(
                                classifier1.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.classifier1.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses1),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
                classifier1.train()
        scheduler1.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_fusion(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData1(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData1(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model1 = ModelExtraction(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        0.15,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    model2 = ModelExtraction(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        0.3,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    model3 = ModelExtraction(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        0.9,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    classifier1 = Classifier1()
    classifier2 = Classifier1()
    classifier3 = Classifier1()
    if use_cuda:
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()
        classifier1 = classifier1.cuda()
        classifier2 = classifier2.cuda()
        classifier3 = classifier3.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion1 = nn.CrossEntropyLoss(weight=weight_rank)
    criterion2 = infonce.InfoNCE()
    if args.optim_type == "Adam":
        optimizer1 = torch.optim.Adam(
            [
                {"params": model1.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )
        optimizer2 = torch.optim.Adam(
            [
                {"params": model2.parameters()},
                {"params": model3.parameters()},
                {"params": classifier2.parameters()},
                {"params": classifier3.parameters()},
            ],
            lr=args.lr,
        )
    elif args.optim_type == "RMSprop":
        optimizer1 = torch.optim.RMSprop(
            [
                {"params": model1.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
        )
        optimizer2 = torch.optim.RMSprop(
            [
                {"params": model2.parameters()},
                {"params": model3.parameters()},
                {"params": classifier2.parameters()},
                {"params": classifier3.parameters()},
            ],
            lr=args.lr,
        )
    elif args.optim_type == "SGD":
        optimizer1 = torch.optim.SGD(
            [
                {"params": model1.parameters()},
                {"params": classifier1.parameters()},
            ],
            lr=args.lr,
            momentum=0.8,
        )
        optimizer2 = torch.optim.SGD(
            [
                {"params": model2.parameters()},
                {"params": model3.parameters()},
                {"params": classifier2.parameters()},
                {"params": classifier3.parameters()},
            ],
            lr=args.lr,
            momentum=0.8,
        )
    else:
        raise ValueError("optim_type is not right!")
    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.1)
    scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model1.train()
    model2.train()
    model3.train()
    classifier1.train()
    classifier2.train()
    classifier3.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels = (
                sfeatures
            )
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()

            # Forward pass
            combine_outputs1 = model1(
                kmer, base_means, base_stds, base_signal_lens, signals
            )
            outputs1, _ = classifier1(combine_outputs1)
            combine_outputs2 = model2(
                kmer.detach(),
                base_means.detach(),
                base_stds.detach(),
                base_signal_lens.detach(),
                signals.detach(),
            )
            outputs2, _ = classifier2(combine_outputs2)
            combine_outputs3 = model3(
                kmer.detach(),
                base_means.detach(),
                base_stds.detach(),
                base_signal_lens.detach(),
                signals.detach(),
            )
            outputs3, _ = classifier3(combine_outputs3)
            loss1 = criterion1(outputs1, labels)
            loss1_2 = criterion1(outputs2, labels)
            loss1_3 = criterion1(outputs3, labels)
            loss2 = criterion2(combine_outputs1, combine_outputs2, combine_outputs3)

            tlosses.append(loss1.detach().item())

            # Backward and optimize
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss1.backward(retain_graph=True)
            loss1_2.backward(retain_graph=True)
            loss1_3.backward(retain_graph=True)
            loss2.backward()

            torch.nn.utils.clip_grad_norm_(model1.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier1.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier2.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(classifier3.parameters(), 0.5)
            optimizer1.step()
            optimizer2.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model1.eval()
                model2.eval()
                model3.eval()
                classifier1.eval()
                classifier2.eval()
                classifier3.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                        vcombine_outputs = model1(
                            vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals
                        )
                        voutputs, vlogits = classifier1(vcombine_outputs)
                        vloss = criterion1(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model1.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model1.train()
                model2.train()
                model3.train()
                classifier1.train()
                classifier2.train()
                classifier3.train()
        scheduler1.step()
        scheduler2.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_cnn(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData1(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData1(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelCNN(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            _, kmer, base_means, base_stds, base_signal_lens, signals, labels = (
                sfeatures
            )
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()

            # Forward pass
            outputs, logits = model(
                kmer, base_means, base_stds, base_signal_lens, signals
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                        voutputs, vlogits = model(
                            vkmer, vbase_means, vbase_stds, vbase_signal_lens, vsignals
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_cg(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData4(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData4(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelCG(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            (
                _,
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                labels,
                tags,
                cg_contents,
            ) = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                tags = tags.cuda()
                cg_contents = cg_contents.cuda()

            # Forward pass
            outputs, logits = model(
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                tags,
                cg_contents,
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vtags,
                            vcg_contents,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vtags = vtags.cuda()
                            vcg_contents = vcg_contents.cuda()
                        voutputs, vlogits = model(
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vtags,
                            vcg_contents,
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def train_combine(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData5(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData5(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelCombine(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            (
                _,
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                labels,
                tags,
                cg_contents,
            ) = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                tags = tags.cuda()
                cg_contents = cg_contents.cuda()

            # Forward pass
            outputs, logits = model(
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                tags,
                cg_contents,
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vtags,
                            vcg_contents,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vtags = vtags.cuda()
                            vcg_contents = vcg_contents.cuda()
                        voutputs, vlogits = model(
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vtags,
                            vcg_contents,
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def trainFreq(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData6(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData6(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelFrequency(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            (
                _,
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                labels,
                signals_freq,
            ) = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                signals_freq = signals_freq.cuda()

            # Forward pass
            outputs, logits = model(
                kmer, base_means, base_stds, base_signal_lens, signals, signals_freq
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vsignals_freq,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vsignals_freq = vsignals_freq.cuda()
                        voutputs, vlogits = model(
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vsignals_freq,
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def trainFreq_mp(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    train_dataset = SignalFeaData6(args.train_file)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )

    valid_dataset = SignalFeaData6(args.valid_file)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(
                r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*"
            )
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelFrequency_mp(
        args.seq_len,
        args.signal_len,
        args.layernum1,
        args.layernum2,
        args.class_num,
        args.dropout_rate,
        args.hid_rnn,
        args.n_vocab,
        args.n_embed,
        str2bool(args.is_base),
        str2bool(args.is_signallen),
        str2bool(args.is_trace),
        args.model_type,
    )
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            (
                _,
                kmer,
                base_means,
                base_stds,
                base_signal_lens,
                signals,
                labels,
                magnitude,
                phase,
            ) = sfeatures
            if use_cuda:
                kmer = kmer.cuda()
                base_means = base_means.cuda()
                base_stds = base_stds.cuda()
                base_signal_lens = base_signal_lens.cuda()
                # base_probs = base_probs.cuda()
                signals = signals.cuda()
                labels = labels.cuda()
                magnitude = magnitude.cuda()
                phase = phase.cuda()

            # Forward pass
            outputs, logits = model(
                kmer, base_means, base_stds, base_signal_lens, signals, magnitude, phase
            )
            loss = criterion(outputs, labels)
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        (
                            _,
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vlabels,
                            vmagnitude,
                            vphase,
                        ) = vsfeatures
                        if use_cuda:
                            vkmer = vkmer.cuda()
                            vbase_means = vbase_means.cuda()
                            vbase_stds = vbase_stds.cuda()
                            vbase_signal_lens = vbase_signal_lens.cuda()
                            # vbase_probs = vbase_probs.cuda()
                            vsignals = vsignals.cuda()
                            vlabels = vlabels.cuda()
                            vmagnitude = vmagnitude.cuda()
                            vphase = vphase.cuda()
                        voutputs, vlogits = model(
                            vkmer,
                            vbase_means,
                            vbase_stds,
                            vbase_signal_lens,
                            vsignals,
                            vmagnitude,
                            vphase,
                        )
                        vloss = criterion(voutputs, vlabels)

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(
                            vlabels.numpy(), vpredicted
                        )
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(
                                model.state_dict(),
                                model_dir
                                + args.model_type
                                + ".b{}_s{}_epoch{}.ckpt".format(
                                    args.seq_len, args.signal_len, epoch + 1
                                ),
                            )
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                no_best_model = False

                    time_cost = time.time() - start
                    print(
                        "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(tlosses),
                            np.mean(vlosses),
                            np.mean(vaccus),
                            np.mean(vprecs),
                            np.mean(vrecas),
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    # model input
    parser.add_argument(
        "--model_type",
        type=str,
        default="both_bilstm",
        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
        required=False,
        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
        "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=21,
        required=False,
        help="len of kmer. default 21",
    )
    parser.add_argument(
        "--signal_len",
        type=int,
        default=16,
        required=False,
        help="the number of signals of one base to be used in deepsignal_plant, default 16",
    )

    # model param
    parser.add_argument(
        "--layernum1",
        type=int,
        default=3,
        required=False,
        help="lstm layer num for combined feature, default 3",
    )
    parser.add_argument(
        "--layernum2",
        type=int,
        default=1,
        required=False,
        help="lstm layer num for seq feature (and for signal feature too), default 1",
    )
    parser.add_argument("--class_num", type=int, default=2, required=False)
    parser.add_argument("--dropout_rate", type=float, default=0.5, required=False)
    parser.add_argument(
        "--n_vocab",
        type=int,
        default=16,
        required=False,
        help="base_seq vocab_size (15 base kinds from iupac)",
    )
    parser.add_argument(
        "--n_embed", type=int, default=4, required=False, help="base_seq embedding_size"
    )
    parser.add_argument(
        "--is_base",
        type=str,
        default="yes",
        required=False,
        help="is using base features in seq model, default yes",
    )
    parser.add_argument(
        "--is_signallen",
        type=str,
        default="yes",
        required=False,
        help="is using signal length feature of each base in seq model, default yes",
    )
    parser.add_argument(
        "--is_trace",
        type=str,
        default="no",
        required=False,
        help="is using trace (base prob) feature of each base in seq model, default yes",
    )

    # BiLSTM model param
    parser.add_argument(
        "--hid_rnn",
        type=int,
        default=256,
        required=False,
        help="BiLSTM hidden_size for combined feature",
    )

    # model training
    parser.add_argument(
        "--optim_type",
        type=str,
        default="Adam",
        choices=["Adam", "RMSprop", "SGD"],
        required=False,
        help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop', default Adam",
    )
    parser.add_argument("--batch_size", type=int, default=512, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument(
        "--max_epoch_num",
        action="store",
        default=20,
        type=int,
        required=False,
        help="max epoch num, default 20",
    )
    parser.add_argument(
        "--min_epoch_num",
        action="store",
        default=5,
        type=int,
        required=False,
        help="min epoch num, default 5",
    )
    parser.add_argument("--step_interval", type=int, default=100, required=False)

    parser.add_argument("--pos_weight", type=float, default=1.0, required=False)
    # parser.add_argument('--seed', type=int, default=1234,
    #                     help='random seed')

    # else
    parser.add_argument("--tmpdir", type=str, default="/tmp", required=False)
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
        "--freq",
        action="store_true",
        default=False,
        help="weather use frequency attribute",
    )

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    if args.transfer:
        train_transfer(args)
    elif args.domain:
        train_domain(args)
    elif args.freq:
        trainFreq(args)
    else:
        train(args)
        # train_fusion(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == "__main__":
    main()
