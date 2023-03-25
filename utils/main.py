# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
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
from datasets import get_dataset
from models import get_model
from utils.training import train_random
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime
import random
import datasets.random_setting

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

def main(args=None):

    lecun_fix()
    if args is None:
        args = parse_args()
    leep_seq = []
    leep_buffer_seq = []
    # octe_seq = []
    logme_seq = []
    logme_buffer_seq = []
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

    sequence_label =  [[[5, 8, 3], [2, 4, 6]], [[6, 2, 1], [0, 7, 3]], [[3, 5, 0], [6, 2, 8]], [[8, 2, 1], [2, 1, 7]], [[2, 1, 4], [1, 2, 3]], [[4, 2, 8], [4, 3, 0]], [[4, 0, 2], [7, 4, 0]], [[8, 1, 2], [2, 6, 0]], [[5, 4, 0], [2, 7, 5]], [[6, 0, 4], [7, 0, 8]], [[1, 6, 0], [5, 7, 8]], [[2, 4, 0], [2, 7, 4]], [[0, 6, 1], [0, 5, 8]], [[5, 6, 4], [6, 8, 7]], [[5, 7, 3], [7, 2, 6]], [[1, 7, 2], [8, 3, 6]], [[4, 8, 0], [5, 2, 1]], [[1, 8, 3], [4, 0, 8]], [[6, 4, 3], [8, 4, 3]], [[7, 6, 0], [7, 0, 5]], [[2, 8, 7], [7, 8, 0]], [[2, 4, 5], [8, 5, 3]], [[8, 6, 3], [3, 8, 7]], [[1, 6, 3], [5, 2, 4]], [[8, 3, 4], [4, 8, 7]], [[8, 7, 5], [5, 8, 4]], [[0, 4, 1], [4, 0, 6]], [[0, 3, 6], [0, 3, 4]], [[6, 2, 4], [1, 4, 2]], [[3, 0, 6], [2, 7, 5]], [[5, 0, 1], [7, 0, 3]], [[3, 1, 6], [8, 5, 2]], [[1, 8, 4], [6, 1, 5]], [[6, 3, 7], [1, 2, 7]], [[1, 5, 8], [6, 0, 4]], [[8, 4, 0], [1, 7, 6]], [[3, 2, 4], [5, 4, 2]], [[3, 2, 8], [3, 8, 6]], [[5, 7, 4], [4, 3, 6]], [[5, 0, 4], [3, 4, 6]], [[2, 6, 8], [8, 4, 0]], [[7, 4, 0], [1, 4, 2]], [[0, 1, 8], [4, 0, 1]], [[3, 2, 1], [3, 1, 6]], [[1, 2, 0], [7, 5, 6]], [[5, 1, 8], [3, 7, 5]], [[0, 7, 3], [2, 8, 5]], [[3, 7, 5], [7, 3, 8]], [[8, 6, 4], [4, 6, 0]], [[5, 6, 0], [8, 7, 0]], [[1, 3, 2], [1, 4, 7]], [[4, 0, 1], [0, 6, 8]], [[8, 4, 7], [0, 3, 6]], [[6, 5, 4], [4, 5, 6]], [[5, 0, 7], [4, 7, 2]], [[0, 2, 3], [8, 5, 0]], [[7, 5, 2], [7, 5, 6]], [[0, 5, 8], [2, 3, 8]], [[8, 2, 1], [2, 3, 0]], [[4, 6, 3], [1, 3, 0]], [[2, 1, 8], [1, 0, 3]], [[5, 6, 4], [2, 5, 4]], [[6, 8, 1], [2, 6, 7]], [[5, 1, 6], [1, 2, 5]], [[3, 5, 7], [6, 7, 3]], [[7, 2, 1], [3, 5, 8]], [[8, 6, 7], [6, 4, 2]], [[4, 2, 8], [3, 6, 0]], [[6, 8, 4], [3, 8, 1]], [[8, 5, 6], [7, 2, 0]], [[8, 6, 2], [6, 5, 1]], [[0, 6, 8], [1, 5, 3]], [[1, 7, 8], [2, 3, 6]], [[2, 4, 3], [5, 0, 6]], [[0, 8, 6], [6, 3, 1]], [[0, 1, 7], [6, 8, 4]], [[6, 5, 4], [3, 8, 5]], [[5, 6, 8], [7, 2, 5]], [[3, 8, 5], [2, 6, 5]], [[5, 8, 6], [0, 8, 3]], [[5, 7, 1], [7, 4, 0]], [[0, 2, 7], [7, 0, 5]], [[7, 3, 1], [3, 0, 2]], [[0, 3, 8], [5, 3, 6]], [[7, 2, 8], [4, 3, 0]], [[2, 7, 3], [8, 1, 3]], [[3, 5, 7], [1, 2, 7]], [[3, 1, 0], [4, 1, 3]], [[3, 4, 6], [4, 7, 6]], [[1, 6, 4], [4, 3, 6]], [[6, 4, 0], [4, 8, 6]], [[2, 3, 7], [5, 1, 0]], [[0, 5, 2], [8, 3, 4]], [[8, 6, 7], [8, 1, 2]], [[4, 8, 1], [4, 5, 6]], [[3, 7, 5], [5, 6, 1]], [[0, 4, 5], [8, 2, 4]], [[1, 4, 5], [3, 0, 2]], [[5, 8, 0], [3, 1, 6]], [[5, 1, 8], [2, 7, 6]]]
    sequence_sample = [[[1200, 900, 1200], [300, 1200, 900]], [[300, 1200, 1200], [600, 300, 600]], [[300, 300, 600], [1200, 600, 300]], [[300, 1200, 600], [600, 1200, 600]], [[300, 1200, 600], [600, 600, 1200]], [[1200, 300, 900], [900, 600, 300]], [[600, 900, 900], [600, 1200, 1200]], [[900, 300, 1200], [1200, 300, 900]], [[600, 1200, 300], [1200, 600, 300]], [[600, 300, 600], [300, 1200, 900]], [[300, 600, 600], [900, 900, 300]], [[600, 300, 300], [600, 1200, 300]], [[600, 1200, 1200], [900, 900, 900]], [[300, 600, 300], [1200, 600, 300]], [[1200, 300, 1200], [900, 600, 300]], [[1200, 900, 900], [600, 600, 1200]], [[300, 300, 1200], [900, 600, 900]], [[1200, 600, 1200], [300, 900, 1200]], [[900, 300, 1200], [600, 1200, 600]], [[1200, 1200, 300], [600, 1200, 1200]], [[300, 1200, 300], [1200, 900, 600]], [[1200, 600, 1200], [900, 1200, 1200]], [[600, 1200, 300], [900, 600, 900]], [[1200, 600, 900], [600, 1200, 1200]], [[600, 300, 600], [900, 1200, 900]], [[300, 1200, 600], [1200, 300, 600]], [[1200, 900, 900], [1200, 900, 1200]], [[600, 900, 300], [1200, 600, 600]], [[1200, 1200, 900], [900, 900, 1200]], [[900, 300, 600], [1200, 300, 300]], [[900, 1200, 900], [600, 900, 900]], [[900, 1200, 600], [600, 1200, 300]], [[900, 600, 300], [300, 900, 600]], [[300, 1200, 1200], [600, 900, 600]], [[300, 600, 900], [600, 600, 300]], [[600, 1200, 1200], [1200, 1200, 1200]], [[600, 900, 300], [900, 1200, 900]], [[300, 1200, 300], [600, 300, 900]], [[600, 900, 300], [600, 1200, 900]], [[300, 600, 600], [900, 900, 900]], [[900, 900, 600], [600, 900, 1200]], [[1200, 300, 1200], [900, 1200, 600]], [[1200, 600, 900], [900, 300, 300]], [[600, 300, 600], [600, 600, 900]], [[300, 600, 900], [900, 300, 600]], [[600, 300, 300], [600, 600, 300]], [[900, 1200, 900], [1200, 600, 300]], [[1200, 900, 900], [900, 900, 600]], [[300, 1200, 900], [300, 300, 600]], [[900, 900, 900], [600, 900, 1200]], [[300, 600, 600], [600, 900, 1200]], [[300, 1200, 900], [900, 900, 300]], [[900, 300, 300], [600, 600, 1200]], [[1200, 300, 300], [1200, 900, 300]], [[600, 300, 300], [900, 900, 1200]], [[300, 600, 300], [1200, 300, 300]], [[600, 600, 600], [600, 900, 600]], [[600, 600, 300], [900, 1200, 1200]], [[300, 300, 900], [300, 600, 1200]], [[300, 600, 1200], [300, 300, 900]], [[300, 600, 300], [600, 1200, 1200]], [[900, 1200, 600], [300, 600, 600]], [[1200, 900, 300], [300, 1200, 300]], [[300, 300, 600], [600, 300, 1200]], [[300, 600, 300], [1200, 300, 600]], [[300, 900, 900], [900, 1200, 600]], [[300, 300, 1200], [1200, 900, 900]], [[900, 900, 600], [1200, 600, 300]], [[900, 1200, 900], [1200, 600, 300]], [[1200, 900, 1200], [900, 600, 300]], [[1200, 600, 600], [900, 1200, 600]], [[1200, 300, 600], [900, 600, 600]], [[1200, 300, 600], [900, 600, 300]], [[1200, 1200, 600], [600, 300, 300]], [[300, 900, 1200], [300, 1200, 300]], [[300, 300, 300], [900, 900, 600]], [[900, 300, 600], [1200, 300, 300]], [[1200, 1200, 900], [900, 900, 300]], [[300, 900, 600], [300, 600, 1200]], [[600, 300, 300], [1200, 900, 300]], [[300, 300, 1200], [300, 600, 600]], [[300, 1200, 1200], [300, 1200, 600]], [[900, 300, 600], [600, 600, 900]], [[300, 300, 300], [1200, 1200, 1200]], [[1200, 600, 300], [900, 300, 600]], [[600, 300, 300], [600, 300, 600]], [[600, 300, 1200], [1200, 1200, 900]], [[1200, 900, 900], [600, 900, 900]], [[1200, 1200, 600], [900, 600, 1200]], [[900, 300, 300], [900, 900, 900]], [[900, 600, 600], [1200, 600, 300]], [[1200, 600, 600], [1200, 900, 1200]], [[600, 300, 1200], [300, 900, 600]], [[1200, 300, 300], [300, 300, 300]], [[900, 900, 600], [900, 300, 600]], [[1200, 1200, 900], [900, 1200, 600]], [[1200, 1200, 600], [1200, 1200, 1200]], [[600, 1200, 900], [600, 900, 300]], [[600, 300, 600], [1200, 900, 600]], [[900, 1200, 600], [600, 600, 900]]]

    print("THIS IS", datasets.random_setting.SETTING, "SETTING")
    for s in range(args.num_seq):
        print("SEQUENCE:", s)


        datasets.random_setting.random_label_list = []
        datasets.random_setting.random_N_SAMPLES_PER_CLASS = []

        
        datasets.random_setting.unique_label_list = []
        datasets.random_setting.count_unique_label_list = []
        datasets.random_setting.sequence_sample_list = []

        random_sample = sequence_label[0][0]

        for label in random_sample:
            if label not in datasets.random_setting.unique_label_list:
                datasets.random_setting.unique_label_list.append(label)
        datasets.random_setting.count_unique_label_list.append(len(datasets.random_setting.unique_label_list))

        datasets.random_setting.random_label_list.append(random_sample)
        
        random_sample_list = sequence_sample[0][0]
        # random_sample_list = []
        # for c in range(datasets.random_setting.random_N_CLASSES_PER_TASK):
        #     random_sample_list.append(random.choice([300, 600, 900, 1200]))
        #     # random_sample_list.append(random.choice([1200]))

        datasets.random_setting.random_N_SAMPLES_PER_CLASS.append(random_sample_list)
        datasets.random_setting.sequence_sample_list.append(sum(random_sample_list))

        for t in range(1, datasets.random_setting.random_N_TASKS):
            while 1:
                random_sample = random.sample(range(0, 9), datasets.random_setting.random_N_CLASSES_PER_TASK)
                flag = 0
                count = 0
                for label in random_sample:
                    if label in datasets.random_setting.unique_label_list:
                        count = count + 1
                if count == datasets.random_setting.random_N_CLASSES_PER_TASK - datasets.random_setting.random_N_CLASSES_PER_TASK:
                    flag = 1
                    break
            # random_sample = sequence_label[s][t]
            # random_sample = random.sample(range(0, 9), datasets.random_setting.random_N_CLASSES_PER_TASK)

            for label in random_sample:
                if label not in datasets.random_setting.unique_label_list:
                    datasets.random_setting.unique_label_list.append(label)
            datasets.random_setting.count_unique_label_list.append(len(datasets.random_setting.unique_label_list))

            datasets.random_setting.random_label_list.append(random_sample)
            
            random_sample_list = sequence_sample[s][t]
            # random_sample_list = []
            # for c in range(datasets.random_setting.random_N_CLASSES_PER_TASK):
            #     random_sample_list.append(random.choice([300, 600, 900, 1200]))
            #     # random_sample_list.append(random.choice([1200]))

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
        dataset = get_dataset(args)

        if args.n_epochs is None and isinstance(dataset, ContinualDataset):
            args.n_epochs = dataset.get_epochs()
        if args.batch_size is None:
            args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
            args.minibatch_size = dataset.get_minibatch_size()

        backbone = dataset.get_backbone()
        loss = dataset.get_loss()
        model = get_model(args, backbone, loss, dataset.get_transform())
        
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))     

        if isinstance(dataset, ContinualDataset):
            full_accs, bwt, total_forget, acc, complex_1epoch, complex_25epoch, complex_50epoch, leep, leep_buffer, logme, logme_buffer, leep_1stacc, bwt_leep = train_random(model, dataset, args)
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
            # octe_seq.append(octe)
            # logme_arxiv_seq.append(logme_arxiv)
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
            # print("octe = ", octe_seq)
            # print("logme_arxiv = ", logme_arxiv_seq)
            print("leep_1stacc = ", leep_1stacc_seq)
            print("bwt_leep = ", bwt_leep_seq)
            print("sequence_label = ", sequence_label_seq)
            print("sequence_sample =", sequence_sample_seq)
            print("sequence_unique =", sequence_unique_seq)
        else:
            assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
            ctrain(args)

        del backbone
        del loss
        del model

        # ------------------------------------------------

        """first_task_label = datasets.random_setting.random_label_list[0]
        first_task_sample = datasets.random_setting.random_N_SAMPLES_PER_CLASS[0]

        second_task_label = datasets.random_setting.random_label_list[1]
        second_task_sample = datasets.random_setting.random_N_SAMPLES_PER_CLASS[1]

        datasets.random_setting.random_label_list = []
        datasets.random_setting.random_N_SAMPLES_PER_CLASS = []

        
        datasets.random_setting.unique_label_list = []
        datasets.random_setting.count_unique_label_list = []
        datasets.random_setting.sequence_sample_list = []

        for t in range(datasets.random_setting.random_N_TASKS):
            while 1:
                random_sample = random.sample(range(0, 9), datasets.random_setting.random_N_CLASSES_PER_TASK)
                flag = 0
                count = 0
                for label in random_sample:
                    if label in datasets.random_setting.unique_label_list:
                        count = count + 1
                if count == datasets.random_setting.random_N_CLASSES_PER_TASK - datasets.random_setting.random_N_CLASSES_PER_TASK:
                    flag = 1
                    break
            # random_sample = random.sample(range(0, 9), datasets.random_setting.random_N_CLASSES_PER_TASK)
            for label in random_sample:
                if label not in datasets.random_setting.unique_label_list:
                    datasets.random_setting.unique_label_list.append(label)
            datasets.random_setting.count_unique_label_list.append(len(datasets.random_setting.unique_label_list))

            datasets.random_setting.random_label_list.append(random_sample)

            random_sample_list = []
            for c in range(datasets.random_setting.random_N_CLASSES_PER_TASK):
                random_sample_list.append(random.choice([300, 600, 900, 1200]))
                # random_sample_list.append(random.choice([1200]))

            datasets.random_setting.random_N_SAMPLES_PER_CLASS.append(random_sample_list)
            datasets.random_setting.sequence_sample_list.append(sum(random_sample_list))"""
        
        # for t in range(1):
        #     random_sample = second_task_label
        #     for label in random_sample:
        #         if label not in datasets.random_setting.unique_label_list:
        #             datasets.random_setting.unique_label_list.append(label)
        #     datasets.random_setting.count_unique_label_list.append(len(datasets.random_setting.unique_label_list))

        #     datasets.random_setting.random_label_list.append(random_sample)

        #     datasets.random_setting.random_N_SAMPLES_PER_CLASS.append(second_task_sample)
        #     datasets.random_setting.sequence_sample_list.append(sum(second_task_sample))





if __name__ == '__main__':
    main()
