# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders_random
from datasets.transforms.denormalization import DeNormalize
from typing import Tuple

import datasets.random_setting

from torch.utils.data import Dataset

class TensorMNIST(Dataset):
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

class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class RandomMNIST(ContinualDataset):

    NAME = 'random-mnist'
    SETTING = datasets.random_setting.SETTING
    N_TASKS = datasets.random_setting.random_N_TASKS
    N_CLASSES_PER_TASK = datasets.random_setting.random_N_CLASSES_PER_TASK
    # HEAD = 'single' #multi or single
    TRANSFORM = None

    def get_data_loaders(self):
        transform = transforms.ToTensor()
        # train_dataset = MyMNIST(base_path() + 'MNIST',
        #                         train=True, download=True, transform=transform)
        train_dataset = TensorMNIST(datasets.random_setting.img_train_tensor,
                                      datasets.random_setting.target_train_tensor,
                                      datasets.random_setting.not_aug_img_train_tensor, train = True)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = TensorMNIST(datasets.random_setting.img_test_tensor,
                                          datasets.random_setting.target_test_tensor,
                                          datasets.random_setting.not_aug_img_test_tensor, train = False)
            # test_dataset = MNIST(base_path() + 'MNIST',
            #                     train=False, download=True, transform=transform)

        train, test = store_masked_loaders_random(train_dataset, test_dataset, 
                                datasets.random_setting.random_label_list,
                                datasets.random_setting.random_N_SAMPLES_PER_CLASS,
                                datasets.random_setting.unique_label_list, self)
        return train, test

    def get_backbone(self):
        if self.SETTING == 'class-il':
            return MNISTMLP(28 * 28, len(datasets.random_setting.unique_label_list))
            # return MNISTMLP(28 * 28, RandomMNIST.N_TASKS
                        # * RandomMNIST.N_CLASSES_PER_TASK)
        return MNISTMLP(28 * 28, self.N_TASKS
                        * self.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914),
                                (0.2470))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None