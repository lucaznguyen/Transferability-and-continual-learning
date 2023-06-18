import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from typing import Tuple
from PIL import Image

from tqdm import tqdm

from utils.conf import base_path

import numpy as np

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
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img

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

        not_aug_img = original_img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img
    
def get_tensor(dataset: str):

    if dataset == "random-mnist":
        TRANSFORM = transforms.ToTensor()

        TEST_TRANSFORM = transforms.ToTensor()

        NOT_AUG_TRANSFORM = transforms.ToTensor()

        train_set = MyMNIST(base_path() + 'MNIST', train=True,
                                        download=True, transform=None)

        test_set = MNIST(base_path() + 'MNIST', train=False,
                                        download=True, transform=None)
        
        mode = "L"

    if dataset == "random-cifar10":
        TRANSFORM = transforms.Compose(
                                        [transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2470, 0.2435, 0.2615))])

        TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2470, 0.2435, 0.2615))])

        NOT_AUG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

        train_set = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                        download=True, transform=None)

        test_set = CIFAR10(base_path() + 'CIFAR10', train=False,
                                        download=True, transform=None)
        
        mode = "RGB"
    
    img_train_tensor = []
    target_train_tensor = []
    not_aug_img_train_tensor = []

    img_test_tensor = []
    target_test_tensor = []
    not_aug_img_test_tensor = []


    for i, data in tqdm(enumerate(train_set.data)):
        if dataset == "random-mnist":
            img = Image.fromarray(data.detach().numpy(), mode=mode)
        if dataset == "random-cifar10":
            img = Image.fromarray(data, mode=mode)
        original_img = img.copy()
        not_aug_img = NOT_AUG_TRANSFORM(original_img)
        img = TRANSFORM(img)
        img_train_tensor.append(img)
        target_train_tensor.append(train_set.targets[i])
        not_aug_img_train_tensor.append(not_aug_img)

    for i, data in tqdm(enumerate(test_set.data)):
        if dataset == "random-mnist":
            img = Image.fromarray(data.detach().numpy(), mode=mode)
        if dataset == "random-cifar10":
            img = Image.fromarray(data, mode=mode)
        original_img = img.copy()
        not_aug_img = NOT_AUG_TRANSFORM(original_img)
        img = TEST_TRANSFORM(img)
        img_test_tensor.append(img)
        target_test_tensor.append(test_set.targets[i])
        not_aug_img_test_tensor.append(not_aug_img)

    img_train_tensor = np.array(img_train_tensor, dtype = torch.Tensor)
    target_train_tensor = np.array(target_train_tensor, dtype = int)
    not_aug_img_train_tensor = np.array(not_aug_img_train_tensor, dtype = torch.Tensor)

    img_test_tensor = np.array(img_test_tensor, dtype = torch.Tensor)
    target_test_tensor = np.array(target_test_tensor, dtype = int)
    not_aug_img_test_tensor = np.array(not_aug_img_test_tensor, dtype = torch.Tensor)

    return img_train_tensor, target_train_tensor, not_aug_img_train_tensor, img_test_tensor, target_test_tensor, not_aug_img_test_tensor