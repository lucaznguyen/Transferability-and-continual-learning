# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.LogME import LogME
# from utils.octe import compute_coupling, compute_CE

from torch.utils.data import DataLoader

from backbone.MNISTMLP import MNISTMLP
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
import datasets.random_setting
from typing import Tuple
from datasets import get_dataset
import math
import random
import sys
from models import get_model

from itertools import permutations

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    
    # if k>0:
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -20
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -20
    # outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    # outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            #    dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """
    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    N, C_s = pseudo_source_label.shape
    print("N, C_s:", N, C_s)
    target_label = target_label.reshape(-1)
    print("target_label:", target_label)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    print("C_t:", C_t)
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    
    joint_sum = joint.sum(axis=0, keepdims=True)
    print("joint_sum", joint_sum)
    p_target_given_source = (joint / joint_sum).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    print("prob: ", empirical_prob)
    return leep_score

# def init_zero_2d_arr(m, n):
#   arr = []
#   for i in range(m):
#     ar = []
#     for j in range(n):
#       ar.append(0)
#     arr.append(ar)
#   return arr

# def init_zero_1d_arr(n):
#   arr = []
#   for i in range(n):
#     arr.append(0)
#   return arr

# def LEEP(labels, feature_extract, num_class):
#   import math
#   sum = 0
#   n = len(labels)
#   n_z = len(feature_extract[0])
#   p_yz = init_zero_2d_arr(n_z, num_class)
#   p_z = init_zero_1d_arr(n_z)
#   for z in range(n_z):
#     for i in range(n):
#       p_yz[z][labels[i]]=p_yz[z][labels[i]]+feature_extract[i][z]/n
#       p_z[z] = p_z[z]+feature_extract[i][z]/n

#   for i in range(n):
#     sum_log = 0
#     for z in range(n_z):
#       sum_log = sum_log + p_yz[z][labels[i]]/p_z[z]*feature_extract[i][z]
#     sum = sum + math.log(sum_log)/n
#   sum = sum
#   return sum

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_logme(model: ContinualModel, dataset: ContinualDataset, train_loader: DataLoader, t: int,
              sample_datapoint: list, label_datapoint: list, cal_buffer: False):
    """
    t: task index
    """
    print("calbuffer", cal_buffer)
    
    features_arr = []
    # outputs_arr = []
    labels_arr = []

    logme = LogME(regression=False)

    correct, total = 0.0, 0.0

    # print("sample_datapoint", sample_datapoint)
    # print("label_datapoint", label_datapoint)

    for _, data in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            # print("inputs", inputs)
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            features = model(x = inputs, big_returnt = "features")

            outputs = model(x = inputs, big_returnt = "out")

            # if dataset.SETTING == "task-il":
                # mask_classes(features, dataset, t)
                # mask_classes(outputs, dataset, t)

            
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            for lab in labels:
                labels_arr.append(int(lab))

            for feats in features:
                features_arr.append(feats.cpu().numpy()/np.linalg.norm(feats.cpu().numpy()))

            # for outs in outputs:
                # outputs_arr.append(outs.cpu().numpy())

    if cal_buffer == True:
        with torch.no_grad():
            sample_datapoint_device = sample_datapoint.to(model.device)
            label_datapoint_device = label_datapoint.to(model.device)
            features = model(x = sample_datapoint_device, big_returnt = "features")

            for lab in label_datapoint_device:
                labels_arr.append(int(lab))
            
            for feats in features:
                features_arr.append(feats.cpu().numpy()/np.linalg.norm(feats.cpu().numpy()))

    features_arr = np.array(features_arr)
    # outputs_arr = np.array(outputs_arr)

    # print("len(features_arr)", np.shape(features_arr))
    print("features_arr:", features_arr)
    # print("outputs_arr:", outputs_arr)
    # print("len(outputs_arr)", len(outputs_arr[0]))     
    target_label = np.array(labels_arr)
    print("len(target_label)", target_label)
    print("unique_target_label:", list(set(target_label)))

    if dataset.SETTING == "task-il":
        alternate_label = labels_arr.copy()
        for i in range(len(alternate_label)):
            if alternate_label[i] in list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK)):
                alternate_label[i] = alternate_label[i] - t*dataset.N_CLASSES_PER_TASK
        alternate_label = np.array(alternate_label)

        print("unique_alternate_label:", list(set(alternate_label)))
        print("LogMe score by feature: ", logme.fit(features_arr, alternate_label))

        return logme.fit(features_arr, alternate_label)

        # previous_label = list(np.array(range(dataset.N_CLASSES_PER_TASK))+(t-1)*dataset.N_CLASSES_PER_TASK)
        # perm_list = list(permutations(previous_label))
        # unique_target_label = list(set(target_label))
        # logme_value = []
        # for perm in perm_list:
        #     alternate_label = labels_arr.copy()
        #     # print("perm", list(perm))
        #     # perm = list(perm)
        #     for i in range(len(alternate_label)):
        #         value = alternate_label[i]
        #         # print("index", unique_target_label.index(value))
        #         alternate_label[i] = perm[unique_target_label.index(value)]
        #     alternate_label = np.array(alternate_label)
        #     logme_value.append(logme.fit(features_arr, alternate_label))
            
        # print("LogMe score by feature: ", max(logme_value))

        # return max(logme_value)


    print("LogMe score by feature: ", logme.fit(features_arr, target_label))
    # print("LogMe score by outputs: ", logme.fit(outputs_arr, target_label))
    print("ACCS AT LOGME WITH TRAINING DATA:", correct / total * 100)
                
    return logme.fit(features_arr, target_label)

def get_octe(model: ContinualModel, dataset: ContinualDataset,
             previous_train_loader: DataLoader, current_train_loader: DataLoader, t: int):
    """
    t: task index
    """
    previous_input_arr = []
    current_input_arr = []
    previous_label_arr = []
    pseudo_current_label_arr = []
    for _, data in enumerate(previous_train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            features = model(x = inputs, big_returnt = "features")

            # if dataset.SETTING == "task-il":
            #     mask_classes(inputs, dataset, t)

            for feats in features:
                previous_input_arr.append(feats.cpu().numpy()/np.linalg.norm(feats.cpu().numpy()))

            # print(outputs_arr)
            for lab in labels:
                previous_label_arr.append(int(lab))

    for _, data in enumerate(current_train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)

            if dataset.SETTING == "task-il":
                mask_classes(outputs, dataset, t)

            features = model(x = inputs, big_returnt = "features")

            # if dataset.SETTING == "task-il":
            #     mask_classes(inputs, dataset, t)

            for feats in features:
                current_input_arr.append(feats.cpu().numpy()/np.linalg.norm(feats.cpu().numpy()))

            _, pred = torch.max(outputs.data, 1)
            pred = list(pred.cpu().numpy())
            pseudo_current_label_arr = pseudo_current_label_arr + pred

    previous_input_arr = torch.tensor(np.array(previous_input_arr), dtype=torch.float)
    current_input_arr = torch.tensor(np.array(current_input_arr), dtype=torch.float)
    previous_label_arr = np.array(previous_label_arr)[:,np.newaxis]
    pseudo_current_label_arr = np.array(pseudo_current_label_arr)[:,np.newaxis]

    P, _ = compute_coupling(previous_input_arr, current_input_arr, previous_label_arr, pseudo_current_label_arr)

    # compute the conditonal entropy (ce)
    ce = compute_CE(P, previous_label_arr, pseudo_current_label_arr)

    print("OCTE score: ", ce)
                
    return ce

def get_leep(model: ContinualModel, dataset: ContinualDataset, train_loader: DataLoader, t: int,
             sample_datapoint: list, label_datapoint: list, cal_buffer: False):
    """
    t: task index
    """

    print("calbuffer", cal_buffer)

    outputs_arr = []
    labels_arr = []
    for _, data in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)

            if dataset.SETTING == "task-il":
                mask_classes(outputs, dataset, t-1)

            # count = 0
            for out in outputs:
                # if count == 0:
                #     print("out", out)
                #     print("softmax", softmax(out.cpu().numpy()))
                #     count = 1
                outputs_arr.append(softmax(out.cpu().numpy()))

            # print(outputs_arr)
            for lab in labels:
                labels_arr.append(int(lab))

    if cal_buffer == True:
        with torch.no_grad():
            sample_datapoint_device = sample_datapoint.to(model.device)
            label_datapoint_device = label_datapoint.to(model.device)
            outputs = model(sample_datapoint_device)

            if dataset.SETTING == "task-il":
                mask_classes(outputs, dataset, t-1)
            
            for out in outputs:
                outputs_arr.append(softmax(out.cpu().numpy()))

            for lab in label_datapoint_device:
                labels_arr.append(int(lab))

    # print("LABELS", list(set(labels_arr)))
    
    pseudo_source_label = np.array(outputs_arr)
    target_label = np.array(labels_arr)

    if dataset.SETTING == "task-il":

        alternate_label = labels_arr.copy()
        for i in range(len(alternate_label)):
            if alternate_label[i] in list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK)):
                alternate_label[i] = alternate_label[i] - t*dataset.N_CLASSES_PER_TASK
        alternate_label = np.array(alternate_label)

        print("unique_alternate_label:", list(set(alternate_label)))
        print("LEEP score: ", LEEP(pseudo_source_label, alternate_label))

        return LEEP(pseudo_source_label, alternate_label)

        # return logme.fit(features_arr, alternate_label)

        # previous_label = list(np.array(range(dataset.N_CLASSES_PER_TASK))+(t-1)*dataset.N_CLASSES_PER_TASK)
        # perm_list = list(permutations(previous_label))
        # unique_target_label = list(set(target_label))
        # leep_value = []
        # for perm in perm_list:
        #     alternate_label = labels_arr.copy()
        #     for i in range(len(alternate_label)):
        #         value = alternate_label[i]
        #         alternate_label[i] = perm[unique_target_label.index(value)]
        #     alternate_label = np.array(alternate_label)
        #     leep_value.append(LEEP(pseudo_source_label, alternate_label))
            
        # print("LEEP score: ", max(leep_value))

        # return max(leep_value)

    print("LEEP score: ", LEEP(pseudo_source_label, target_label))
                
    return LEEP(pseudo_source_label, target_label)

def simple_complexity(dataset: ContinualDataset, train_loader: DataLoader, num_epoch: int, t: int):
    
    print("SIMPLE MODEL WITH", num_epoch, "TRAINING AT TASK", t+1)
    
    simple_loss = torch.nn.CrossEntropyLoss()
    if dataset.SETTING == "task-il":
        simple_model = MNISTMLP(28 * 28, dataset.N_CLASSES_PER_TASK)
    if dataset.SETTING == "class-il":
        simple_model = MNISTMLP(28 * 28, datasets.random_setting.count_unique_label_list[t])
    simple_model.net.train()

    optimizer = torch.optim.SGD(simple_model.parameters(), lr= 0.01, weight_decay=0.0001)

    for epoch in range(num_epoch):
        count = 0
        for i, data in enumerate(train_loader):

            inputs, labels, not_aug_inputs = data

            optimizer.zero_grad()
            
            outputs = simple_model(inputs)
            new_labels = []
            if dataset.SETTING == "task-il":
                for ele in labels:
                    new_labels.append(int(ele) - dataset.N_CLASSES_PER_TASK*t)
                labels = torch.LongTensor(new_labels)

            # if count == 0:
            #     count = 1
            #     print("labels:", labels)
            #     print("new labels:", new_labels)


            loss_function = simple_loss(outputs, labels)
            loss_function.backward()
            
            optimizer.step()

            progress_bar(i, len(train_loader), epoch, t, float(loss_function))
                
    simple_accs = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        if k == t:
            correct, total = 0.0, 0.0
            for data in test_loader:
                with torch.no_grad():
                    inputs, labels = data

                    new_labels = []
                    if dataset.SETTING == "task-il":
                        for ele in labels:
                            new_labels.append(int(ele) - dataset.N_CLASSES_PER_TASK*t)
                            labels = torch.LongTensor(new_labels)

                    outputs = simple_model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
            simple_accs  = correct / total * 100
        elif k>t:
            break
    
    del simple_model
    del simple_loss
    
    return 100 - simple_accs
    

def evaluate_random(model: ContinualModel, dataset: ContinualDataset, last=False) -> list:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()

    accs = []

    for k, test_loader in enumerate(dataset.test_loaders):
        # print("k", k)
        if last and k < len(dataset.test_loaders) - 1:
            continue

        #sanity check for correct label
        label_list = []
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                # inputs, labels = inputs.to(model.device), labels.to(model.device
                for lab in labels:
                    label_list.append(int(lab))
        set_label = list(set(label_list))
        count_list = [0] * len(set_label)
        for lab in label_list:
            count_list[set_label.index(lab)] = count_list[set_label.index(lab)] + 1

        print("set_label", set_label)
        print("count_list", count_list)

        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                outputs = model(inputs)

                if dataset.SETTING == "task-il":
                    mask_classes(outputs, dataset, k)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

        accs.append(correct / total * 100)
        # print("accs", accs)
        # print("accs_mask_classes", accs_mask_classes)

    model.net.train(status)
    return accs

def train_random(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    leep_score = []
    leep_buffer_score = []
    octe_score = []
    logme_score = []
    logme_buffer_score = []
    # logme_arxiv_score = []
    complex_list_1epoch = []
    complex_list_25epoch = []
    complex_list_50epoch = [] 

    train_data_list = []

    get_data_sample_from_current_task = []
    get_data_label_from_current_task = []


    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)

    total_complex_1epoch, total_complex_25epoch, total_complex_50epoch = 0, 0, 0

    bwt_leep = 0
    
    forget = 0

    # complete_acc_list = []

    if dataset.SETTING == "class-il":
        print("count_unique_label_list", datasets.random_setting.count_unique_label_list)

    print("PREPARATION PHASE")
    for t in range(dataset.N_TASKS):
        print("TASK", t)
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate_random(model, dataset_copy)

    print(file=sys.stderr)
    print("TRAINING PHASE")

    for t in range(dataset.N_TASKS):
        print("TASK", t)

        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        train_data_list.append(train_loader)

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            if hasattr(model, 'buffer'):
                get_data_sample_from_current_task, get_data_label_from_current_task = model.buffer.get_all_data()[0], model.buffer.get_all_data()[1]
            else:
                get_data_sample_from_current_task = torch.Tensor(get_data_sample_from_current_task)
                get_data_label_from_current_task = torch.Tensor(get_data_label_from_current_task)

            accs = evaluate_random(model, dataset, last=True)
            # octe_score.append(get_octe(model, dataset, train_data_list[t-1], train_data_list[t], t))
            leep_score.append(get_leep(model, dataset, train_data_list[t], t,
                                        get_data_sample_from_current_task,
                                        get_data_label_from_current_task,
                                        cal_buffer = False))
            leep_buffer_score.append(get_leep(model, dataset, train_data_list[t], t,
                                        get_data_sample_from_current_task,
                                        get_data_label_from_current_task,
                                        cal_buffer = True))
            logme_score.append(get_logme(model, dataset, train_data_list[t], t,
                                         get_data_sample_from_current_task,
                                         get_data_label_from_current_task,
                                         cal_buffer = False))
            logme_buffer_score.append(get_logme(model, dataset, train_data_list[t], t,
                                         get_data_sample_from_current_task,
                                         get_data_label_from_current_task,
                                         cal_buffer = True))
            # logme_arxiv_score.append(logme02)
            results[t-1] = results[t-1] + accs

            temp = list(get_data_sample_from_current_task.cpu().numpy())
            get_data_sample_from_current_task = temp

            temp = list(get_data_label_from_current_task.cpu().numpy())
            get_data_label_from_current_task = temp

        scheduler = dataset.get_scheduler(model, args)

        # setup first task for training until converge
        # if t:
        #     range_epoch = model.args.n_epochs
        # else:
        #     range_epoch = 50

        # if t:
        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                inputs, labels, not_aug_inputs = data
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                if len(get_data_sample_from_current_task) < datasets.random_setting.count_unique_label_list[t]:
                    for j in range(len(labels)):
                        if int(labels[j]) not in get_data_label_from_current_task:
                            get_data_label_from_current_task.append(int(labels[j]))
                            get_data_sample_from_current_task.append(inputs[j].cpu().numpy())

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
            
            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if t == 0:
            print("SAVEEEEEEEE")
            model.net.load_state_dict(torch.load("save/task1classil.pt"))

        accs = evaluate_random(model, dataset)
    
        # torch.save(model.net.state_dict(), "save/task"+str(t+1)+"classil.pt")

        print("accs", accs)

        results.append(accs)

        if t:
            forget = forget + forgetting(results)
        # print("forget", forget_list)


        mean_acc = np.mean(accs)

        print("Accuracy for task(s):", t+1, " ", "["+dataset.SETTING+"]",":", mean_acc)

        # 1 epoch
        complex_1epoch = simple_complexity(dataset, train_loader, 1, t)
        complex_list_1epoch.append(complex_1epoch)
        total_complex_1epoch = total_complex_1epoch + complex_1epoch
        print("complex 1 epoch", complex_1epoch)

        #25 epoch
        complex_25epoch = 0#simple_complexity(dataset, train_loader, 25, t)
        complex_list_25epoch.append(complex_25epoch)
        total_complex_25epoch = total_complex_25epoch + complex_25epoch
        print("complex 25 epoch", complex_25epoch)

        #50 epoch
        complex_50epoch = 0#simple_complexity(dataset, train_loader, 50, t)
        complex_list_50epoch.append(complex_50epoch)
        total_complex_50epoch = total_complex_50epoch + complex_50epoch
        print("complex 50 epoch", complex_50epoch)

    # for t in range(dataset.N_TASKS-1):
    #     print(train_data_list)
    #     train_loader = train_data_list[t]
    #     backbone = dataset.get_backbone()
    #     simple_loss = dataset.get_loss()
    #     simple_model = get_model(args, backbone, simple_loss, dataset.get_transform())
    #     simple_model.net.to(model.device)
    #     simple_model.net.train()
    #     scheduler = dataset.get_scheduler(simple_model, args)
    #     for epoch in range(simple_model.args.n_epochs):
    #         count = 0
    #         for i, data in enumerate(train_loader):
    #             if hasattr(dataset.train_loader.dataset, 'logits'):
    #                 inputs, labels, not_aug_inputs = data
    #                 inputs = inputs.to(simple_model.device)
    #                 labels = labels.to(simple_model.device)
    #                 not_aug_inputs = not_aug_inputs.to(simple_model.device)
    #                 logits = logits.to(simple_model.device)
    #                 simple_loss = simple_model.observe(inputs, labels, not_aug_inputs, logits)
    #             else:
    #                 inputs, labels, not_aug_inputs = data
    #                 inputs, labels = inputs.to(simple_model.device), labels.to(
    #                     simple_model.device)
    #                 not_aug_inputs = not_aug_inputs.to(simple_model.device)
    #                 simple_loss = simple_model.observe(inputs, labels, not_aug_inputs)
    #             if not count:
    #                 print(labels)
    #             count = 1
                
    #         if scheduler is not None:
    #             scheduler.step()
    #     bwt_leep = bwt_leep+get_leep(simple_model, dataset, train_data_list[dataset.N_TASKS-1], t)/dataset.N_TASKS

    bwt_leep = 0

    return results, backward_transfer(results), forget, mean_acc/100, total_complex_1epoch/100, total_complex_25epoch/100, total_complex_50epoch/100, np.mean(leep_score), np.mean(leep_buffer_score), np.mean(logme_score), np.mean(logme_buffer_score), np.mean(leep_score)+(100-results[0][0])/100, bwt_leep
