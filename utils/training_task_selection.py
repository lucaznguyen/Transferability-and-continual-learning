# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import random

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
    
def evaluate(model: ContinualModel, dataset: ContinualDataset, max_num_list: list, test_data_list: DataLoader, last=False) -> list:
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

    
    for k, test_loader in enumerate(test_data_list):

        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                if len(data) == 2:
                    inputs, labels = data
                elif len(data) == 3:
                    inputs, labels, _ = data
                # inputs, labels = data

                labels = swap_class_to_current_task(labels, max_num_list[k], dataset.N_CLASSES_PER_TASK, k)

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

def swap_class_to_current_task(labels: list, max_num, count_unique: int, t: int):

    # print("original label_list: ", labels)
    labels_copy = labels.tolist().copy()
    for i in range(len(labels_copy)):
        labels_copy[i] = (max_num-int(labels_copy[i]))+t*count_unique
        
    # print("swapped label_list: ", labels_copy)

    return torch.Tensor(labels_copy)



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

    original_train_data_list = []
    original_test_data_list = []

    original_position = list(range(dataset.N_TASKS))

    train_data_list = []
    test_data_list = []
    max_num_list = []

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

    max_class_current_task_list = []
    position = []

    # print("PREPARATION PHASE")
    for t in range(dataset.N_TASKS):
    #     print("TASK", t)
    #     model.net.train()
        train_loader, test_loader = dataset_copy.get_data_loaders()

        dist = 0

        task_distance.append(dist)

        max_class_current_task_list.append(dataset.N_CLASSES_PER_TASK-1 + t*dataset.N_CLASSES_PER_TASK)
        original_train_data_list.append(train_loader)
        original_test_data_list.append(test_loader)

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
        print("TASK", t+1)

        model.net.train()


        if t == 0:
            train_loader, test_loader = original_train_data_list[0], original_test_data_list[0]
            print(type(train_loader))
            train_data_list.append(train_loader)
            test_data_list.append(test_loader)
            max_num_list.append(max_class_current_task_list[0])
            position.append(original_position[0])
            random_num = 0

        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            if hasattr(model, 'buffer'):
                get_data_sample_from_current_task, get_data_label_from_current_task = model.buffer.get_all_data()[0], model.buffer.get_all_data()[1]
            else:
                get_data_sample_from_current_task = torch.Tensor(get_data_sample_from_current_task)
                get_data_label_from_current_task = torch.Tensor(get_data_label_from_current_task)

            # accs = evaluate(model, dataset, last=True)

            # original_train_data_list = swap_class_to_current_task(original_train_data_list, dataset.N_CLASSES_PER_TASK, t)
            # original_test_data_list = swap_class_to_current_task(original_test_data_list, dataset.N_CLASSES_PER_TASK, t)

            print("Remaining tasks:", original_train_data_list)

            all_logme_future = []
            for j, train_data in enumerate(original_train_data_list):
                need_to_change_list = list(range(max_class_current_task_list[j]-dataset.N_CLASSES_PER_TASK + 1, max_class_current_task_list[j]+1))
                stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))

                print("need_to_change:", need_to_change_list)

                all_logme_future.append(get_logme(model, dataset, train_data, t,
                                        get_data_sample_from_current_task,
                                        get_data_label_from_current_task,
                                        need_to_change_list, stay_list,
                                        cal_buffer = False))
            
            print("LogMe score for these tasks:", all_logme_future)

            if args.task_selection == "random":
                random_num = random.choice(list(range(len(original_train_data_list)))) 
            elif args.task_selection == "best":
                random_num = all_logme_future.index(max(all_logme_future))
            elif args.task_selection == "worst":
                random_num = all_logme_future.index(min(all_logme_future))

            print("Choose num:", random_num)
            
            train_loader, test_loader = original_train_data_list[random_num], original_test_data_list[random_num]

            need_to_change_list = list(range(max_class_current_task_list[random_num]-dataset.N_CLASSES_PER_TASK + 1, max_class_current_task_list[random_num]+1))
            stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))

            train_data_list.append(train_loader)
            test_data_list.append(test_loader)
            max_num_list.append(max_class_current_task_list[random_num])
            position.append(original_position[random_num])
            
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

            # logme_arxiv_score.append(logme02)
            results[t-1] = results[t-1] + [-1]*(dataset.N_TASKS-t)
            print(results)

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

                labels = swap_class_to_current_task(labels, max_class_current_task_list[random_num], dataset.N_CLASSES_PER_TASK, t)

                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                if len(get_data_sample_from_current_task) < datasets.random_setting.count_unique_label_list[t]:
                    for j in range(len(labels)):
                        if int(labels[j]) not in get_data_label_from_current_task:
                            get_data_label_from_current_task.append(int(labels[j]))
                            get_data_sample_from_current_task.append(inputs[j].cpu().numpy())

                if hasattr(train_loader.dataset, 'logits'):
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

        accs = evaluate(model, dataset, max_num_list, test_data_list)
    
        # torch.save(model.net.state_dict(), "save/task"+str(t+1)+"classil35.pt")

        # print("accs", accs)

        results.append(accs)

        if t:
            forget = 0# forget + forgetting(results)
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
            if (sub_t > t):
                continue
            need_to_change_list = list(range(sub_t*dataset.N_CLASSES_PER_TASK, sub_t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK))
            stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))
            # print("3. Calculating REVERSE LOGME CONTINUAL MODEL ON TASK " + str(t+1))
            logme_model_score.append(get_logme(model, dataset, train_data_list[sub_t], sub_t,
                                            get_data_sample_from_current_task,
                                            get_data_label_from_current_task,
                                            need_to_change_list, stay_list,
                                            cal_buffer = False))

        logme_simple_model_score_1epoch = 0
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

        max_class_current_task_list.pop(random_num)
        original_train_data_list.pop(random_num)
        original_test_data_list.pop(random_num)
        original_position.pop(random_num)


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
    bwt = 0#backward_transfer(results)

    return  task_distance,\
            results, bwt, forget, mean_acc/100,\
            complex_list_1epoch, complex_list_25epoch,\
            complex_list_50epoch, leep_score, leep_buffer_score,\
            logme_score, logme_buffer_score,\
            logme_model_list, logme_simple_model_1epoch_list, logme_simple_model_50epoch_list,\
            np.mean(leep_score)+(100-results[0][0])/100, bwt_leep, position
