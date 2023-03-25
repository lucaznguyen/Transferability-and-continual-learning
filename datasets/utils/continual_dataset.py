# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim
import random

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass



def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """


    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]

    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader

def store_masked_loaders_random(train_dataset: datasets, test_dataset: datasets, random_label_list: list,
                    random_N_SAMPLES_PER_CLASS: list, unique_label_list: list,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """

    print("random_label_list", random_label_list)

    print("unique_label_list", unique_label_list)

    train_sample_list = random_N_SAMPLES_PER_CLASS[setting.i].copy()
    
    print("train_sample_list", train_sample_list)

    # print("train_sample_list", train_sample_list)

    train_mask, test_mask = [], []
    for target in train_dataset.targets:
        if target in random_label_list[setting.i]:
            index = random_label_list[setting.i].index(int(target))
            if train_sample_list[index] > 0:
                train_mask.append(True)
                train_sample_list[index] = train_sample_list[index] - 1
                # print(int(target))
            else:
                # print(int(target))
                train_mask.append(False)
        else:
            train_mask.append(False)
            
    for target in test_dataset.targets:
        test_mask.append(target in random_label_list[setting.i])

    train_mask = np.array(train_mask)
    test_mask = np.array(test_mask)


    train_dataset.data = train_dataset.data[train_mask]

    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    print("Label used:", list(set(train_dataset.targets)))

            
    # print("train_sample_list", train_sample_list)
    # for label in random_label_list[setting.i]:
    #     count = 0
    #     for target in train_dataset.targets:
    #         if int(target) == label:
    #             count = count + 1
    #     print("label", label, " count", count)

    if setting.SETTING == 'class-il':
        for i in range(len(train_dataset.targets)):
            index = unique_label_list.index(train_dataset.targets[i])
            train_dataset.targets[i] = index
        
        for i in range(len(test_dataset.targets)):
            index = unique_label_list.index(test_dataset.targets[i])
            test_dataset.targets[i] = index
            
    if setting.SETTING == 'task-il':
        for i in range(len(train_dataset.targets)):
            index = random_label_list[setting.i].index(train_dataset.targets[i])
            train_dataset.targets[i] = setting.i*setting.N_CLASSES_PER_TASK + index
        
        for i in range(len(test_dataset.targets)):
            index = random_label_list[setting.i].index(test_dataset.targets[i])
            test_dataset.targets[i] = setting.i*setting.N_CLASSES_PER_TASK + index

    if setting.SETTING == 'task-il':
        print("Label after task-il masked:", list(set(train_dataset.targets)))

    if setting.SETTING == 'class-il':
        print("Label after class-il masked:", list(set(train_dataset.targets)))

    # mask_label_list = list(set(train_dataset.targets))
    # remainder = n_example - (n_example//setting.N_CLASSES_PER_TASK)*setting.N_CLASSES_PER_TASK
    # n_cls_list = [n_example//setting.N_CLASSES_PER_TASK] * setting.N_CLASSES_PER_TASK
    # n_cls_list[setting.N_CLASSES_PER_TASK - 1] = n_cls_list[setting.N_CLASSES_PER_TASK - 1] + remainder

    # choose_dataset_data = []
    # choose_dataset_target = []

    # for i in range(len(train_dataset.targets)):
    #     index = mask_label_list.index(train_dataset.targets[i])
    #     if n_cls_list[index] > 0:
    #         n_cls_list[index] = n_cls_list[index] - 1
    #         choose_dataset_data.append(train_dataset.data[i])
    #         choose_dataset_target.append(train_dataset.targets[i])
    
    # train_dataset.data = np.array(choose_dataset_data)
    # train_dataset.targets = np.array(choose_dataset_target)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader

def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
