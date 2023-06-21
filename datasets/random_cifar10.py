# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders_random
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

from torch.utils.data import Dataset

import random
import numpy as np

import datasets.random_setting

class TensorCIFAR10(Dataset):
    def __init__(self, img, target, not_aug_img, train = True) -> None:

        self.train = train

        self.data = img
        self.targets = target
        self.not_aug_data = not_aug_img

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        if self.train:
            return self.data[index], self.targets[index], self.not_aug_data[index]
        else:
            return self.data[index], self.targets[index]

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class RandomCIFAR10(ContinualDataset):

    NAME = 'random-cifar10'
    SETTING = datasets.random_setting.SETTING
    N_CLASSES_PER_TASK = datasets.random_setting.random_N_CLASSES_PER_TASK
    N_TASKS = datasets.random_setting.random_N_TASKS
    # HEAD = 'multi' #multi or single
    # N_EXAMPLE = 300
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        # path = "tensor/cifar10/"
        # print("path", path)

        # train_dataset = TensorCIFAR10(datasets.random_setting.img_train_tensor,
        #                               datasets.random_setting.target_train_tensor,
        #                               datasets.random_setting.not_aug_img_train_tensor, train = True)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            # test_dataset = TensorCIFAR10(datasets.random_setting.img_test_tensor,
            #                               datasets.random_setting.target_test_tensor,
            #                               datasets.random_setting.not_aug_img_test_tensor, train = False)
            # test_dataset = TensorCIFAR10(datasets.random_setting.test_tensor, train = False)
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders_random(train_dataset, test_dataset, 
                                datasets.random_setting.random_label_list,
                                datasets.random_setting.random_N_SAMPLES_PER_CLASS,
                                datasets.random_setting.unique_label_list, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), RandomCIFAR10.TRANSFORM])
        return transform

    def get_backbone(self):
        if RandomCIFAR10.SETTING == 'class-il':
            return resnet18(len(datasets.random_setting.unique_label_list))
        return resnet18(self.N_TASKS
                        * self.N_CLASSES_PER_TASK)
    # def get_backbone():
    #     if RandomCIFAR10.HEAD == 'single':
    #         return resnet18(10)
    #     return resnet18(RandomCIFAR10.N_TASKS
    #                     * RandomCIFAR10.N_CLASSES_PER_TASK)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None
