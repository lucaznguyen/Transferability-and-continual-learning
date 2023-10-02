import warnings

import numpy as np
from numba import njit

from backbone.MNISTMLP import MNISTMLP
from backbone.ResNet18 import resnet18
from backbone.MNISTResNet18 import mnist_resnet18

import math
import random

from typing import Tuple

import torch

from torch.optim import SGD
import torch
import torchvision

import datasets.random_setting


from torch.utils.data import DataLoader

from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset

from argparse import Namespace

from utils.conf import get_device
from utils.status import progress_bar

from models import get_model

from itertools import permutations


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # print("alpha", alpha)
        # print("beta", beta)
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


# use pseudo data to compile the function
# D = 20, N = 50
# f_tmp = np.random.randn(20, 50).astype(np.float64)
# each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(), np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50, 20)


@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh
truncated_svd(np.random.randn(20, 10).astype(np.float64))


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms = list(self.ms)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, D = f.shape  # k = min(N, D)
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k
        # s.shape = k
        # vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for k in range(100):
                # print("alpha", alpha)
                # print("beta", beta)
                # print("iter:", k)
                old_alpha = alpha
                old_beta = beta
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N

                if abs(old_alpha-alpha) <=1e-3 and abs(old_beta-beta)<=1e-3:
                    break
                # if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    # break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            # self.ms = list(self.ms)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)
    
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
    
def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """
    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """

    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, target_label)])
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_simple_logme(model: MNISTMLP, dataset: ContinualDataset, train_loader: DataLoader, t: int,
              sample_datapoint: list, label_datapoint: list,
              need_to_change_list: list, stay_list: list,
              cal_buffer: False):
    """
    t: task index
    """
    
    features_arr = []
    labels_arr = []

    logme = LogME(regression=False)

    correct, total = 0.0, 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for _, data in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            # print("inputs", inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            
            features = model(x = inputs, returnt = "features")

            outputs = model(x = inputs, returnt = "out")

            
            _, pred = torch.max(outputs.data, 1)
            pred.to(device)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            for lab in labels:
                labels_arr.append(int(lab))

            for feats in features:
                features_arr.append(feats.cpu().numpy())

    if cal_buffer == True:
        with torch.no_grad():
            if hasattr(model, 'device'):
                sample_datapoint_device = sample_datapoint.to(model.device)
                label_datapoint_device = label_datapoint.to(model.device)
            else:
                sample_datapoint_device = sample_datapoint
                label_datapoint_device = label_datapoint
            features = model(x = sample_datapoint_device, returnt = "features")

            for lab in label_datapoint_device:
                labels_arr.append(int(lab))
            
            for feats in features:
                features_arr.append(feats.cpu().numpy())

    features_arr = np.array(features_arr)  
    target_label = np.array(labels_arr)

    if dataset.SETTING == "task-il":
        alternate_label = labels_arr.copy()
        for i in range(len(alternate_label)):
            if alternate_label[i] in need_to_change_list:
                alternate_label[i] = stay_list[need_to_change_list.index(alternate_label[i])]
        alternate_label = np.array(alternate_label)

        return logme.fit(features_arr, alternate_label)
                
    return logme.fit(features_arr, target_label)

def simple_model_logme_score(args: Namespace, epoch: int, dataset: ContinualDataset, train_data_list: list, t: int,
                get_data_sample_from_current_task: list, get_data_label_from_current_task: list,
                need_to_change_list: list, stay_list: list,
                current_task: int,
                simple_model = None):
    
    if simple_model == None:
        simple_loss = torch.nn.CrossEntropyLoss()

        if args.dataset == "random-mnist-resnet":
            if dataset.SETTING == "task-il":
                simple_model = mnist_resnet18(dataset.N_CLASSES_PER_TASK, grayscale=True)
            if dataset.SETTING == "class-il":
                simple_model = mnist_resnet18(datasets.random_setting.count_unique_label_list[t], grayscale=True)
        elif "mnist" in args.dataset:
        # if args.dataset == "random-mnist" or args.dataset == "random-fashion-mnist":
            if dataset.SETTING == "task-il":
                simple_model = MNISTMLP(28 * 28, dataset.N_CLASSES_PER_TASK)
            if dataset.SETTING == "class-il":
                simple_model = MNISTMLP(28 * 28, datasets.random_setting.count_unique_label_list[t])
        elif "cifar" in args.dataset or "tinyimg" in args.dataset:
            if dataset.SETTING == "task-il":
                simple_model = resnet18(dataset.N_CLASSES_PER_TASK)
            if dataset.SETTING == "class-il":
                simple_model = resnet18(datasets.random_setting.count_unique_label_list[t])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        simple_model = simple_model.to(device)

        optimizer = torch.optim.SGD(simple_model.parameters(), lr= 0.01, weight_decay=0.0001)

        for epoch in range(epoch):
            count = 0
            for i, data in enumerate(train_data_list[dataset.N_TASKS-1]):

                inputs, labels, not_aug_inputs = data
                
                new_labels = []
                new_inputs = []


                if dataset.SETTING == "task-il":
                    accept_labels = list(range(dataset.N_CLASSES_PER_TASK*t, dataset.N_CLASSES_PER_TASK*(t+1)))
                    for j, ele in enumerate(labels):
                        if int(ele) in accept_labels:
                            new_labels.append(int(ele) - dataset.N_CLASSES_PER_TASK*t)
                            new_inputs.append(inputs[j].detach().numpy())
                    
                    if len(new_inputs) == 0:
                        continue

                    inputs = torch.Tensor(np.array(new_inputs))
                    labels = torch.LongTensor(new_labels)

                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = simple_model(inputs)

                loss_function = simple_loss(outputs, labels)
                loss_function.backward()
                
                optimizer.step()

                progress_bar(i, len(train_data_list[dataset.N_TASKS-1]), epoch, t, float(loss_function))

    logme_simple_model_score = []

    for t in range(dataset.N_TASKS):
        # print("4. Calculating REVERSE LOGME SIMPLE MODEL " + str(epoch) + " EPOCH ON TASK" + str(t+1))
        need_to_change_list = list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK))
        # stay_list = need_to_change_list.copy()
        stay_list = list(range(0, dataset.N_CLASSES_PER_TASK))
        logme_simple_model_score.append(get_simple_logme(simple_model, dataset, train_data_list[t], t,
                                         get_data_sample_from_current_task,
                                         get_data_label_from_current_task,
                                         need_to_change_list, stay_list,
                                         cal_buffer = False))
        
    del simple_model
    return logme_simple_model_score

def get_logme(model: ContinualModel, dataset: ContinualDataset, train_loader: DataLoader, t: int,
              sample_datapoint: list, label_datapoint: list,
              need_to_change_list: list, stay_list: list,
              cal_buffer: False):
    """
    t: task index
    """
    # print("calbuffer", cal_buffer)
    
    features_arr = []
    # outputs_arr = []
    labels_arr = []

    logme = LogME(regression=False)

    correct, total = 0.0, 0.0

    for _, data in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            if hasattr(model, 'device'):
                inputs, labels = inputs.to(model.device), labels.to(model.device)
            features = model(x = inputs, big_returnt = "features")

            outputs = model(x = inputs, big_returnt = "out")

            
            _, pred = torch.max(outputs.data, 1)
            if hasattr(model, 'device'):
                pred = pred.to(model.device)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            for lab in labels:
                labels_arr.append(int(lab))

            for feats in features:
                features_arr.append(feats.cpu().numpy())


    if cal_buffer == True:
        with torch.no_grad():
            if hasattr(model, 'device'):
                sample_datapoint_device = sample_datapoint.to(model.device)
                label_datapoint_device = label_datapoint.to(model.device)
            else:
                sample_datapoint_device = sample_datapoint
                label_datapoint_device = label_datapoint
            features = model(x = sample_datapoint_device, big_returnt = "features")

            for lab in label_datapoint_device:
                labels_arr.append(int(lab))
            
            for feats in features:
                features_arr.append(feats.cpu().numpy())

    features_arr = np.array(features_arr)
    # outputs_arr = np.array(outputs_arr)

    # print("len(features_arr)", np.shape(features_arr))
    # print("features_arr:", features_arr)
    # print("outputs_arr:", outputs_arr)
    # print("len(outputs_arr)", len(outputs_arr[0]))     
    target_label = np.array(labels_arr)
    # print("len(target_label)", target_label)
    # print("unique_target_label:", list(set(target_label)))

    if dataset.SETTING == "task-il":
        alternate_label = labels_arr.copy()
        if cal_buffer != True:
            for i in range(len(alternate_label)):
                if alternate_label[i] in need_to_change_list:
                    alternate_label[i] = stay_list[need_to_change_list.index(alternate_label[i])]
        alternate_label = np.array(alternate_label)

        return logme.fit(features_arr, alternate_label)

                
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

    ce = compute_CE(P, previous_label_arr, pseudo_current_label_arr)
      
    return ce

def get_leep(model: ContinualModel, dataset: ContinualDataset, train_loader: DataLoader, t: int,
             sample_datapoint: list, label_datapoint: list, cal_buffer: False):
    """
    t: task index
    """

    outputs_arr = []
    labels_arr = []
    for _, data in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels, _ = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)

            if dataset.SETTING == "task-il":
                mask_classes(outputs, dataset, t-1)

            for out in outputs:
                outputs_arr.append(softmax(out.cpu().numpy()))

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

    pseudo_source_label = np.array(outputs_arr)
    target_label = np.array(labels_arr)

    if dataset.SETTING == "task-il":

        alternate_label = labels_arr.copy()
        for i in range(len(alternate_label)):
            if alternate_label[i] in list(range(t*dataset.N_CLASSES_PER_TASK, t*dataset.N_CLASSES_PER_TASK+dataset.N_CLASSES_PER_TASK)):
                alternate_label[i] = alternate_label[i] - t*dataset.N_CLASSES_PER_TASK
        alternate_label = np.array(alternate_label)

        return LEEP(pseudo_source_label, alternate_label)
                
    return LEEP(pseudo_source_label, target_label)

def simple_complexity(args: Namespace, dataset: ContinualDataset, train_loader: DataLoader, num_epoch: int, t: int):
    
    print("SIMPLE MODEL WITH", num_epoch, "EPOCH AT TASK", t+1)
    
    simple_loss = torch.nn.CrossEntropyLoss()

    if args.dataset == "random-mnist-resnet":
        if dataset.SETTING == "task-il":
            simple_model = mnist_resnet18(dataset.N_CLASSES_PER_TASK, grayscale=True)
        if dataset.SETTING == "class-il":
            simple_model = mnist_resnet18(datasets.random_setting.count_unique_label_list[t], grayscale=True)
    elif "mnist" in args.dataset:
        if dataset.SETTING == "task-il":
            simple_model = MNISTMLP(28 * 28, dataset.N_CLASSES_PER_TASK)
        if dataset.SETTING == "class-il":
            simple_model = MNISTMLP(28 * 28, datasets.random_setting.count_unique_label_list[t])
    elif "cifar" in args.dataset or "tinyimg" in args.dataset:
        # print("ok")
        if dataset.SETTING == "task-il":
            simple_model = resnet18(dataset.N_CLASSES_PER_TASK)
            # print("ok nua roi")
        if dataset.SETTING == "class-il":
            simple_model = resnet18(datasets.random_setting.count_unique_label_list[t])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simple_model = simple_model.to(device)

    optimizer = torch.optim.SGD(simple_model.parameters(), lr= 0.01, weight_decay=0.0001)

    for epoch in range(num_epoch):
        count = 0
        for i, data in enumerate(train_loader):

            inputs, labels, not_aug_inputs = data
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            
            new_labels = []
            new_inputs = []
            if dataset.SETTING == "task-il":
                accept_labels = list(range(dataset.N_CLASSES_PER_TASK*t, dataset.N_CLASSES_PER_TASK*(t+1)))
                for j, ele in enumerate(labels):
                    if int(ele) in accept_labels:
                        new_labels.append(int(ele) - dataset.N_CLASSES_PER_TASK*t)
                        new_inputs.append(inputs[j].detach().numpy())
                
                if len(new_inputs) == 0:
                    continue

                inputs = torch.Tensor(np.array(new_inputs))
                labels = torch.LongTensor(new_labels)

            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = simple_model(inputs)

            # labels = labels.to(device)

            loss_function = simple_loss(outputs, labels)
            loss_function.backward()
            
            optimizer.step()

            progress_bar(i, len(train_loader), epoch, t, float(loss_function))
    
    print("\n")
    
    simple_accs = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        if k == t:
            correct, total = 0.0, 0.0
            for data in test_loader:
                with torch.no_grad():
                    if len(data) == 2:
                        inputs, labels = data
                    elif len(data) == 3:
                        inputs, labels, _ = data
                    # inputs, labels = data

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    new_labels = []
                    if dataset.SETTING == "task-il":
                        for ele in labels:
                            new_labels.append(int(ele) - dataset.N_CLASSES_PER_TASK*t)
                            labels = torch.LongTensor(new_labels)

                    labels = labels.to(device)

                    outputs = simple_model(inputs)

                _, pred = torch.max(outputs.data, 1)
                pred = pred.to(device)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
            simple_accs  = correct / total * 100
        elif k>t:
            break
    
    # del simple_model
    del simple_loss
    
    return 100 - simple_accs, simple_model