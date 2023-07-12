# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
import pandas as pd
import ast
import importlib
import os
import inspect
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES


from models import get_all_models

from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from utils.training import train
from utils.best_args import best_args
from utils.augmentations import *
from utils.conf import set_random_seed

import setproctitle
import torch
import uuid
import datetime
import datasets.random_setting

from utils.tensor import get_tensor

import numpy as np

import torch

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    # parser.add_argument('--seq_num', type=int, required=True,
    #                     help='Sequence amount.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'        
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args

def string_list_to_list(df, col):
    check = df[col].to_list()
    for i in range(len(check)):
        check[i] = ast.literal_eval(check[i])
    return check

def main(args=None):

    lecun_fix()
    if args is None:
        args = parse_args()

    benchmark_dir = "benchmark/"+str(args.n_task_per_seq)+" task/1200 sample/"+args.case+".csv"

    datasets.random_setting.random_N_TASKS = args.n_task_per_seq
    datasets.random_setting.random_N_CLASSES_PER_TASK = args.n_class_per_task

    df = pd.read_csv(benchmark_dir)
    
    datasets.random_setting.sequence_label = string_list_to_list(df, "label")
    datasets.random_setting.sequence_sample = string_list_to_list(df, "sample")
    
    # if args.model not in ["bic", "lucir", "icarl"]:
    #     datasets.random_setting.img_train_tensor,\
    #     datasets.random_setting.target_train_tensor,\
    #     datasets.random_setting.not_aug_img_train_tensor,\
    #     datasets.random_setting.img_test_tensor,\
    #     datasets.random_setting.target_test_tensor,\
    #     datasets.random_setting.not_aug_img_test_tensor = get_tensor(args.dataset)

    leep_seq = []
    leep_buffer_seq = []
    # octe_seq = []
    logme_seq = []
    logme_buffer_seq = []
    logme_model_seq = []
    logme_simple_model_1epoch_seq = []
    logme_simple_model_50epoch_seq = []
    # logme_arxiv_seq = []
    full_accs_seq = []
    acc_seq = []
    bwt_seq = []
    complex_1epoch_seq = []
    complex_25epoch_seq = []
    complex_50epoch_seq = []
    forget_seq = []
    leep_1stacc_seq = []
    bwt_leep_seq = []
    sequence_sample_seq = []
    sequence_label_seq = []
    sequence_unique_seq = []

    # sequence_label = [[[8, 7, 3], [2, 5, 6]], [[1, 3, 6], [2, 5, 4]], [[5, 4, 0], [3, 7, 1]], [[7, 1, 6], [0, 5, 3]], [[0, 6, 2], [4, 1, 7]], [[1, 7, 8], [0, 4, 2]], [[6, 7, 3], [0, 5, 8]], [[4, 8, 3], [0, 5, 2]], [[4, 5, 3], [2, 7, 1]], [[7, 0, 2], [6, 8, 5]], [[6, 4, 8], [5, 1, 7]], [[7, 1, 6], [5, 2, 8]], [[5, 7, 3], [1, 6, 2]], [[6, 7, 5], [0, 1, 8]], [[5, 1, 0], [6, 2, 4]], [[3, 1, 6], [8, 2, 5]], [[4, 7, 3], [2, 5, 6]], [[1, 3, 2], [4, 7, 8]], [[4, 2, 1], [3, 0, 6]], [[1, 3, 0], [2, 8, 4]], [[2, 3, 8], [1, 4, 0]], [[7, 5, 3], [0, 1, 4]], [[4, 1, 7], [5, 6, 8]], [[2, 6, 1], [4, 3, 5]], [[3, 6, 8], [2, 5, 7]], [[6, 3, 4], [0, 1, 8]], [[1, 3, 2], [8, 6, 7]], [[8, 2, 1], [6, 0, 5]], [[4, 3, 1], [2, 7, 5]], [[7, 3, 6], [0, 4, 2]], [[5, 8, 2], [4, 6, 1]], [[2, 6, 5], [1, 7, 8]], [[4, 8, 3], [2, 0, 5]], [[7, 4, 2], [0, 6, 1]], [[7, 3, 8], [4, 6, 2]], [[2, 8, 0], [1, 6, 3]], [[4, 2, 1], [5, 8, 7]], [[7, 6, 5], [0, 8, 1]], [[6, 7, 3], [2, 8, 1]], [[7, 1, 5], [3, 0, 6]], [[3, 4, 2], [1, 8, 5]], [[7, 6, 3], [1, 0, 2]], [[6, 4, 2], [1, 3, 5]], [[8, 2, 0], [7, 5, 4]], [[5, 4, 8], [2, 3, 1]], [[3, 1, 7], [0, 2, 4]], [[6, 4, 5], [3, 1, 7]], [[6, 0, 8], [3, 5, 7]], [[7, 3, 8], [4, 6, 5]], [[3, 0, 8], [4, 2, 7]], [[3, 4, 8], [5, 7, 0]], [[5, 6, 7], [8, 1, 3]], [[7, 6, 3], [0, 4, 1]], [[6, 5, 4], [1, 3, 8]], [[3, 2, 4], [0, 8, 7]], [[4, 1, 2], [6, 7, 3]], [[1, 3, 0], [2, 8, 6]], [[4, 1, 2], [0, 5, 3]], [[1, 7, 2], [4, 6, 5]], [[6, 3, 0], [5, 1, 7]], [[2, 5, 8], [3, 0, 1]], [[7, 5, 6], [4, 8, 1]], [[8, 6, 2], [1, 4, 7]], [[5, 1, 0], [2, 8, 4]], [[4, 0, 1], [8, 7, 2]], [[2, 3, 6], [4, 5, 0]], [[4, 7, 1], [2, 8, 5]], [[6, 4, 2], [3, 7, 0]], [[4, 5, 1], [8, 6, 3]], [[3, 8, 4], [7, 1, 6]], [[1, 7, 5], [3, 6, 2]], [[3, 4, 8], [2, 0, 6]], [[7, 0, 5], [1, 2, 4]], [[2, 7, 1], [0, 6, 4]], [[1, 2, 4], [0, 8, 5]], [[5, 1, 6], [2, 8, 4]], [[1, 0, 3], [6, 2, 4]], [[1, 7, 5], [6, 4, 0]], [[5, 1, 8], [6, 0, 4]], [[2, 5, 8], [3, 1, 0]], [[7, 1, 0], [4, 2, 5]], [[2, 4, 3], [5, 0, 8]], [[6, 2, 0], [1, 3, 8]], [[2, 3, 7], [4, 6, 5]], [[0, 1, 2], [4, 8, 3]], [[3, 4, 2], [1, 6, 8]], [[2, 0, 4], [1, 5, 3]], [[7, 5, 3], [1, 6, 0]], [[1, 4, 5], [6, 0, 2]], [[3, 4, 6], [1, 0, 5]], [[0, 7, 1], [5, 8, 4]], [[2, 7, 8], [4, 1, 5]], [[5, 6, 3], [4, 7, 0]], [[4, 7, 5], [2, 0, 8]], [[0, 6, 7], [3, 1, 2]], [[0, 7, 3], [4, 5, 6]], [[0, 4, 2], [1, 5, 7]], [[6, 7, 1], [4, 5, 8]], [[1, 4, 3], [7, 8, 0]], [[1, 0, 6], [7, 4, 2]]]
    # sequence_sample = [[[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]], [[1200, 1200, 1200], [1200, 1200, 1200]]]
    ###
    sequence_label = datasets.random_setting.sequence_label
    sequence_sample = datasets.random_setting.sequence_sample

    if args.n_epochs == 1:
        scenario = "online"
    else:
        scenario = "offline"

    save_dir = "visualize/"+args.model+"/"+scenario+"/"+str(args.n_task_per_seq)+" task/"+"1200 sample/"+args.case+"/"
    
    if args.drive:
        save_dir = "/content/drive/MyDrive/GitHub/understand-cf/" + save_dir

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    print("THIS IS", datasets.random_setting.SETTING, "SETTING")

    if args.range == '-1':
        lower = 0
        upper = args.num_seq
        if args.num_seq == -1:
            upper = len(sequence_label)
    else:
        sequence_range = ast.literal_eval(args.range)
        lower, upper = sequence_range[0], sequence_range[1]


    flag = 1

    if args.range == '-1':
        name = datasets.random_setting.SETTING+"-"+\
                str(args.n_epochs)+"epoch-"+\
                str(datasets.random_setting.random_N_TASKS)+"task-"+\
                str(int((args.buffer_size/3)*100/1200))+"pbuffer" +\
                ".csv"
    else:
        name = datasets.random_setting.SETTING+"-"+\
                str(args.n_epochs)+"epoch-"+\
                str(datasets.random_setting.random_N_TASKS)+"task-"+\
                str(int((args.buffer_size/3)*100/1200))+"pbuffer"+\
                str(lower)+str(upper)+\
                ".csv"

    for s in range(lower, upper):
        print("SEQUENCE:", s)

        datasets.random_setting.random_label_list = []
        datasets.random_setting.random_N_SAMPLES_PER_CLASS = []

        datasets.random_setting.unique_label_list = []
        datasets.random_setting.count_unique_label_list = []
        datasets.random_setting.sequence_sample_list = []

        for t in range(datasets.random_setting.random_N_TASKS):
            random_sample = sequence_label[s][t]

            for label in random_sample:
                if label not in datasets.random_setting.unique_label_list:
                    datasets.random_setting.unique_label_list.append(label)
            datasets.random_setting.count_unique_label_list.append(len(datasets.random_setting.unique_label_list))

            datasets.random_setting.random_label_list.append(random_sample)
            
            random_sample_list = sequence_sample[s][t]

            datasets.random_setting.random_N_SAMPLES_PER_CLASS.append(random_sample_list)
            datasets.random_setting.sequence_sample_list.append(sum(random_sample_list))

        if datasets.random_setting.SETTING == "task-il":
            datasets.random_setting.count_unique_label_list = []
            for t in range(datasets.random_setting.random_N_TASKS):
                datasets.random_setting.count_unique_label_list.append((t+1)*datasets.random_setting.random_N_CLASSES_PER_TASK)
        

        # Add uuid, timestamp and hostname for logging
        args.conf_jobnum = str(uuid.uuid4())
        args.conf_timestamp = str(datetime.datetime.now())
        args.conf_host = socket.gethostname()

        while 1:
            try:

                from datasets import get_dataset

                DATASET_NAMES[args.dataset].N_TASKS = datasets.random_setting.random_N_TASKS
                DATASET_NAMES[args.dataset].SETTING = datasets.random_setting.SETTING
                DATASET_NAMES[args.dataset].N_CLASSES_PER_TASK = datasets.random_setting.random_N_CLASSES_PER_TASK
                
                dataset = get_dataset(args)
                print("NUM TASK:", dataset.N_TASKS)
                # dataset.N_TASKS = datasets.random_setting.random_N_TASKS
                # dataset.SETTING = datasets.random_setting.SETTING
                # dataset.N_CLASSES_PER_TASK = datasets.random_setting.random_N_CLASSES_PER_TASK

                from models import get_model

                if args.n_epochs is None and isinstance(dataset, ContinualDataset):
                    args.n_epochs = dataset.get_epochs()
                if args.batch_size is None:
                    args.batch_size = dataset.get_batch_size()
                if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
                    args.minibatch_size = dataset.get_minibatch_size()

                backbone = dataset.get_backbone()
                loss = dataset.get_loss()
                model = get_model(args, backbone, loss, dataset.get_transform())

                # if args.model == "bic":
                #     model.n_tasks = datasets.random_setting.random_N_TASKS
                #     model.cpt = datasets.random_setting.random_N_CLASSES_PER_TASK

                if args.model == "xder":
                    model.cpt = datasets.random_setting.random_N_CLASSES_PER_TASK
                    model.tasks = datasets.random_setting.random_N_TASKS

                    denorm = dataset.get_denormalization_transform()
                    model.dataset_mean, model.dataset_std = denorm.mean, denorm.std

                    model.dataset_shape = dataset.get_data_loaders()[0].dataset.data.shape[2]
                    model.gpu_augmentation = strong_aug(model.dataset_shape, model.dataset_mean, model.dataset_std)

                    dataset = get_dataset(args)
                
                # set job name
                setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))     

                if isinstance(dataset, ContinualDataset):
                    full_accs, bwt, total_forget, acc,\
                    complex_1epoch, complex_25epoch, complex_50epoch,\
                    leep, leep_buffer,\
                    logme, logme_buffer, logme_model,\
                    logme_simple_model_1epoch, logme_simple_model_50epoch,\
                    leep_1stacc, bwt_leep = train(model, dataset, args)

                    full_accs_seq.append(full_accs)
                    acc_seq.append(acc)
                    bwt_seq.append(bwt)
                    forget_seq.append(total_forget)
                    complex_1epoch_seq.append(complex_1epoch)
                    complex_25epoch_seq.append(complex_25epoch)
                    complex_50epoch_seq.append(complex_50epoch)
                    leep_seq.append(leep)
                    leep_buffer_seq.append(leep_buffer)
                    logme_seq.append(logme)
                    logme_buffer_seq.append(logme_buffer)
                    logme_model_seq.append(logme_model)
                    logme_simple_model_1epoch_seq.append(logme_simple_model_1epoch)
                    logme_simple_model_50epoch_seq.append(logme_simple_model_50epoch)
                    leep_1stacc_seq.append(leep_1stacc)
                    bwt_leep_seq.append(bwt_leep)
                    sequence_label_seq.append(datasets.random_setting.random_label_list)
                    sequence_sample_seq.append(datasets.random_setting.random_N_SAMPLES_PER_CLASS)
                    sequence_unique_seq.append(datasets.random_setting.count_unique_label_list)
                    
                    print("bwt = ", bwt_seq)
                    print("forget = ", forget_seq)
                    print("full_accs = ", full_accs_seq)
                    print("acc = ", acc_seq)
                    print("complex_1epoch = ", complex_1epoch_seq)
                    print("complex_25epoch = ", complex_25epoch_seq)
                    print("complex_50epoch = ", complex_50epoch_seq)
                    print("leep = ", leep_seq)
                    print("leep_buffer = ", leep_buffer_seq)
                    print("logme = ", logme_seq)
                    print("logme_buffer = ", logme_buffer_seq)
                    print("logme_model = ", logme_model_seq)
                    print("logme_simple_model_1epoch = ", logme_simple_model_1epoch_seq)
                    print("logme_simple_model_50epoch = ", logme_simple_model_50epoch_seq)
                    print("leep_1stacc = ", leep_1stacc_seq)
                    print("bwt_leep = ", bwt_leep_seq)
                    print("sequence_label = ", sequence_label_seq)
                    print("sequence_sample =", sequence_sample_seq)
                    print("sequence_unique =", sequence_unique_seq)

                    full = [[bwt_seq[i], forget_seq[i], full_accs_seq[i],
                                acc_seq[i], complex_1epoch_seq[i], complex_25epoch_seq[i], complex_50epoch_seq[i],
                                leep_seq[i], leep_buffer_seq[i],
                                logme_seq[i], logme_buffer_seq[i],
                                logme_model_seq[i], logme_simple_model_1epoch_seq[i], logme_simple_model_50epoch_seq[i],
                                leep_1stacc_seq[i], bwt_leep_seq[i],
                                sequence_label_seq[i], sequence_sample_seq[i], sequence_unique_seq[i]]
                                for i in range(s+1)]
                    col = ["bwt", "forget", "full_accs",
                        "acc", "complex_1epoch", "complex_25epoch", "complex_50epoch",
                        "leep", "leep_buffer",
                        "logme", "logme_buffer",
                        "logme_model", "logme_simple_model_1epoch", "logme_simple_model_50epoch",
                        "leep_1stacc", "bwt_leep",
                        "sequence_label", "sequence_sample", "sequence_unique"]
                    
                    df = pd.DataFrame(data=full, columns=col)
                    if args.num_seq <= 100:
                        try:
                            df.to_csv(save_dir + name, index=False)
                        except:
                            if flag:
                                state = s
                                flag = 0
                            print("Failed to save drive at sequence", state)
                            df.to_csv("/content/" + name, index=False)
                    else:
                        if s%50 == 0:
                            try:
                                df.to_csv(save_dir + name, index=False)
                            except:
                                if flag:
                                    state = s
                                    flag = 0
                                print("Failed to save drive at sequence", state)
                                df.to_csv("/content/" + name, index=False)

                else:
                    assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
                    ctrain(args)

                del backbone
                del loss
                del model
                
                break

            except Exception as err:
                print("Error:", err)
                print("=> Run again")
                continue



if __name__ == '__main__':
    main()
