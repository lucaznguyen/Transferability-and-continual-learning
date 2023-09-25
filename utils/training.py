# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.transferability import get_leep, get_logme, simple_complexity, simple_model_logme_score

from torch.utils.data import DataLoader

from utils.status import progress_bar
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
import datasets.random_setting
from datasets import get_dataset
import sys

# from scipy.spatial import distance_matrix
import pandas as pd

import torch.nn as nn

def distance_matrix(A, B):
    dist = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            dist = dist + abs(A[i][j]-B[i][j])
    return dist

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
    
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> list:
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
        if last and k < len(dataset.test_loaders) - 1:
            continue

        #sanity check for correct label
        label_list = []
        for data in test_loader:
            with torch.no_grad():
                # print(data)
                # print("LEN:", len(data))
                if len(data) == 2:
                    inputs, labels = data
                elif len(data) == 3:
                    inputs, labels, _ = data
                # inputs, labels = inputs.to(model.device), labels.to(model.device
                for lab in labels:
                    label_list.append(int(lab))
        set_label = list(set(label_list))
        count_list = [0] * len(set_label)
        for lab in label_list:
            count_list[set_label.index(lab)] = count_list[set_label.index(lab)] + 1

        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                if len(data) == 2:
                    inputs, labels = data
                elif len(data) == 3:
                    inputs, labels, _ = data
                # inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                outputs = model(inputs)

                if dataset.SETTING == "task-il":
                    mask_classes(outputs, dataset, k)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

        accs.append(correct / total * 100)

    model.net.train(status)
    return accs

def linear_probing(model: ContinualModel, dataset: ContinualDataset,
                   train_loader: DataLoader, test_loader: DataLoader, t: int,
                   k: int, epoch: int) -> float:
    """
    model: Continual model
    dataset: Current benchmark
    train_loader: Training data
    test_loader: Testing data
    t: current task index
    k: example per class (low)
    epoch: num of epoch
    """

    linear_probing_model = nn.Sequential(
        nn.Linear(100, dataset.N_CLASSES_PER_TASK)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_probing_model.parameters(), lr=0.001)

    dict_k_sample = {}
    for i in range(len(dataset.N_CLASSES_PER_TASK)):
        dict_k_sample[str(i)] = 0
    list_k_input = []
    list_k_label = []

    # print("dict:", dict_k_sample)
    
    for _, data in enumerate(train_loader):
        flag = 1
        inputs, labels, _ = data
        labels = labels - t*dataset.N_CLASSES_PER_TASK
        if dict_k_sample[str(labels)]<k:
            dict_k_sample[str(labels)] += 1
            list_k_input.append(inputs)
            list_k_label.append(labels)
        
        for key in dict_k_sample.keys:
            if dict_k_sample[key]<k:
                flag = 0
                break
        if flag == 1:
            break

    # Train the model
    for epoch in range(epoch):
        for i in range(len(list_k_input)):
        # for _, data in enumerate(train_loader):
            inputs, labels = list_k_input[i], list_k_label[i]
            # print("inputs", inputs)
            if hasattr(model, 'device'):
                inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()
            features = model(x = inputs, big_returnt = "features")
            outputs = linear_probing_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            if len(data) == 2:
                inputs, labels = data
            elif len(data) == 3:
                inputs, labels, _ = data
            inputs, labels = data

            features = model(x = inputs, big_returnt = "features")
            labels = labels - t*dataset.N_CLASSES_PER_TASK

            outputs = linear_probing_model(features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # print("acc k shot using "+str(epoch)+":", accuracy)

    del linear_probing_model
    del criterion
    del optimizer

    return accuracy

def train(model: ContinualModel, dataset: ContinualDataset,
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

    logme_model_list = []
    logme_simple_model_1epoch_list = []
    logme_simple_model_50epoch_list = []
    
    linear_probing_score_1epoch = []
    linear_probing_score_50epoch = []

    complex_list_1epoch = []
    complex_list_25epoch = []
    complex_list_50epoch = [] 

    train_data_list = []

    get_data_sample_from_current_task = []
    get_data_label_from_current_task = []

    task_distance = []


    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)

    total_complex_1epoch, total_complex_25epoch, total_complex_50epoch = 0, 0, 0

    bwt_leep = 0
    
    forget = 0

    # complete_acc_list = []

    # if dataset.SETTING == "class-il":
        # print("count_unique_label_list", datasets.random_setting.count_unique_label_list)

    # print("PREPARATION PHASE")
    for t in range(dataset.N_TASKS):
    #     print("TASK", t)
    #     model.net.train()
        train_loader, _ = dataset_copy.get_data_loaders()

        if dataset.SETTING == "task-il":
            cluster_list = list(range(t*dataset.N_CLASSES_PER_TASK, (t+1)*dataset.N_CLASSES_PER_TASK))
        else:
            cluster_list = list(range(0, 10))
        
        # cluster_list = [str(cluster) for cluster in cluster_list]
        
        df = dict()
        for cluster in cluster_list:
            df[cluster] = []
        
        for i, data in enumerate(train_loader):
            inputs, label_list, _ = data
            label_list = label_list.type(torch.LongTensor)
            for j, label in enumerate(label_list):
                df[int(label)].append(np.array(inputs[j]))

        for cluster in cluster_list:
            df[cluster] = np.array(df[cluster])
            df[cluster] = np.array(np.mean(df[cluster], axis = 0))
            # print(np.array(df[cluster]).shape)
        
        dist = 0
        for i in range(len(cluster_list)-1):
            for j in range(i+1, len(cluster_list)):
                
                dist = dist + distance_matrix(df[cluster_list[i]][0], df[cluster_list[j]][0])

        task_distance.append(dist)

        train_data_list.append(train_loader)

    if args.train_log == 0:
        return task_distance, results, backward_transfer(results), forget, 0,\
            complex_list_1epoch, complex_list_25epoch,\
            complex_list_50epoch, leep_score, leep_buffer_score,\
            logme_score, logme_buffer_score,\
            logme_model_list, logme_simple_model_1epoch_list, logme_simple_model_50epoch_list,\
            0, bwt_leep

    for t in range(dataset.N_TASKS):

        logme_model_score = []
        logme_simple_model_score_1epoch = []
        logme_simple_model_score_50epoch = []

        torch.cuda.empty_cache()
        print("TASK", t)

        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        # train_data_list.append(train_loader)

        # print("model parammmmmmmmmmmmm:")
        # for p in model.parameters():
        #     print(p)

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            if hasattr(model, 'buffer'):
                get_data_sample_from_current_task, get_data_label_from_current_task = model.buffer.get_all_data()[0], model.buffer.get_all_data()[1]
            else:
                get_data_sample_from_current_task = torch.Tensor(get_data_sample_from_current_task)
                get_data_label_from_current_task = torch.Tensor(get_data_label_from_current_task)

            accs = evaluate(model, dataset, last=True)
            # octe_score.append(get_octe(model, dataset, train_data_list[t-1], train_data_list[t], t))

            need_to_change_list = list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK))
            stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))
            
            leep_score.append(get_leep(model, dataset, train_data_list[t], t,
                                        get_data_sample_from_current_task,
                                        get_data_label_from_current_task,
                                        cal_buffer = False))
            leep_buffer_score.append(get_leep(model, dataset, train_data_list[t], t,
                                        get_data_sample_from_current_task,
                                        get_data_label_from_current_task,
                                        cal_buffer = True))
            # print("1. Calculating LOGME FROM TASK " + str(t) + " TO TASK " + str(t+1))
            logme_score.append(get_logme(model, dataset, train_data_list[t], t,
                                         get_data_sample_from_current_task,
                                         get_data_label_from_current_task,
                                         need_to_change_list, stay_list,
                                         cal_buffer = False))
            # print("2. Calculating LOGME-BUFFER FROM TASK " + str(t) + " TO TASK " + str(t+1))
            logme_buffer_score.append(get_logme(model, dataset, train_data_list[t], t,
                                         get_data_sample_from_current_task,
                                         get_data_label_from_current_task,
                                         need_to_change_list, stay_list,
                                         cal_buffer = True))
            # linear_probing_score_1epoch.append(linear_probing(model, dataset,
                                                            #   train_loader, test_loader, t,
                                                            #   5, 1))
            # linear_probing_score_50epoch.append(linear_probing(model, dataset,
                                                            #   train_loader, test_loader, t,
                                                            #   5, 50))
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
                labels = labels.type(torch.LongTensor)
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

        # if t == 0:
            # print("SAVEEEEEEEE")
            # model.net.load_state_dict(torch.load("save/task1taskil.pt"))

        accs = evaluate(model, dataset)
    
        # torch.save(model.net.state_dict(), "save/task"+str(t+1)+"classil35.pt")

        # print("accs", accs)

        results.append(accs)

        if t:
            forget = forget + forgetting(results)
        # print("forget", forget_list)


        mean_acc = np.mean(accs)

        print("\n")
        print("Accuracy for task(s):", t+1, " ", "["+dataset.SETTING+"]",":", mean_acc)

        #1 epoch
        complex_1epoch, simple_model_1epoch = simple_complexity(args, dataset, train_loader, 1, t)
        complex_list_1epoch.append(complex_1epoch)
        total_complex_1epoch = total_complex_1epoch + complex_1epoch
        print("Simple Complexity 1 epoch:", complex_1epoch)

        #25 epoch
        complex_25epoch = 0#simple_complexity(args, dataset, train_loader, 25, t)
        complex_list_25epoch.append(complex_25epoch)
        total_complex_25epoch = total_complex_25epoch + complex_25epoch

        print("Simple Complexity 25 epoch:", complex_25epoch)

        #50 epoch
        if args.offline_complexity:
            complex_50epoch, simple_model_50epoch = simple_complexity(args, dataset, train_loader, 50, t)
        else:
            complex_50epoch = 0
        complex_list_50epoch.append(complex_50epoch)
        total_complex_50epoch = total_complex_50epoch + complex_50epoch
        print("Simple Complexity 50 epoch:", complex_50epoch)

        for sub_t in range(dataset.N_TASKS):
        # for sub_t in range(t + 1):
            need_to_change_list = list(range(sub_t*dataset.N_CLASSES_PER_TASK, sub_t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK))
            stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))
            # print("3. Calculating REVERSE LOGME CONTINUAL MODEL ON TASK " + str(t+1))
            logme_model_score.append(get_logme(model, dataset, train_data_list[sub_t], sub_t,
                                            get_data_sample_from_current_task,
                                            get_data_label_from_current_task,
                                            need_to_change_list, stay_list,
                                            cal_buffer = False))

        logme_simple_model_score_1epoch = simple_model_logme_score(args, 1, dataset, train_data_list, t,
                    get_data_sample_from_current_task, get_data_label_from_current_task,
                    need_to_change_list, stay_list,
                    dataset.N_TASKS,
                    simple_model = simple_model_1epoch)
        if args.offline_logme:
            logme_simple_model_score_50epoch = simple_model_logme_score(args, 50, dataset, train_data_list, t,
                        get_data_sample_from_current_task, get_data_label_from_current_task,
                        need_to_change_list, stay_list,
                        dataset.N_TASKS,
                        simple_model = simple_model_50epoch)
        else:
            logme_simple_model_score_50epoch = 0

        logme_model_list.append(logme_model_score)
        logme_simple_model_1epoch_list.append(logme_simple_model_score_1epoch)
        logme_simple_model_50epoch_list.append(logme_simple_model_score_50epoch)



    # for t in range(dataset.N_TASKS):
    #     need_to_change_list = list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK))
    #     stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))
    #     # print("3. Calculating REVERSE LOGME CONTINUAL MODEL ON TASK " + str(t+1))
    #     logme_model_score.append(get_logme(model, dataset, train_data_list[t], t,
    #                                      get_data_sample_from_current_task,
    #                                      get_data_label_from_current_task,
    #                                      need_to_change_list, stay_list,
    #                                      cal_buffer = False))

    # logme_simple_model_score_1epoch = simple_model_logme_score(args, 1, dataset, train_data_list, t,
    #             get_data_sample_from_current_task, get_data_label_from_current_task,
    #             need_to_change_list, stay_list)
    # if args.offline_logme:
    #     logme_simple_model_score_50epoch = simple_model_logme_score(args, 50, dataset, train_data_list, t,
    #                 get_data_sample_from_current_task, get_data_label_from_current_task,
    #                 need_to_change_list, stay_list)
    # else:
    #     logme_simple_model_score_50epoch = 0

    bwt_leep = 0

    return  task_distance,\
            results, backward_transfer(results), forget, mean_acc/100,\
            complex_list_1epoch, complex_list_25epoch,\
            complex_list_50epoch, leep_score, leep_buffer_score,\
            logme_score, logme_buffer_score,\
            logme_model_list, logme_simple_model_1epoch_list, logme_simple_model_50epoch_list,\
            np.mean(leep_score)+(100-results[0][0])/100, bwt_leep
