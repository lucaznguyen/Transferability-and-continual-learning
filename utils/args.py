# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--num_seq', type=int, default = -1,
                        help='Number of sequence.')
    parser.add_argument('--range', type=str, default = '-1',
                        help='Range of the sequence.')
    parser.add_argument('--case', type=str, required=True,
                        help='Which case to perform experiments on.')
    parser.add_argument('--resume', type=int, default = 0,
                        help='Resume training with remaining sequences.')
    parser.add_argument('--train_log', type = int, default = 1,
                        help='Log the training process and Train')
    parser.add_argument('--n_sample', type=int, default = 1200,
                        help='Number of sample per class.')
    parser.add_argument('--n_class_per_task', type=int, required=True,
                        help='number of class per task.')
    parser.add_argument('--n_task_per_seq', type=int, required=True,
                        help='numer of task per sequence.')
    parser.add_argument('--offline_complexity', type=int, default = 1,
                        help='Calculate complexity 50 epoch.')
    parser.add_argument('--offline_logme', type=int, default = 1,
                        help='Calculate logme simple model 50 epoch.')
    parser.add_argument('--drive', type=int, default = 0,
                        help='Save on drive.')
    parser.add_argument('--task_selection', type = str, default = 'best',
                        help= 'random: randomize the task, best: best task selection, worst: worst task selection')

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default = 32,
                        help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default = 32,
                        help='The batch size of the memory buffer.')
