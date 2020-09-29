from typing import Union
import numpy as np

import torch
from torch.utils.data import TensorDataset

from dnadapt.globals import device


def as_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    """
    Transforms input array and labels into torch tensorDataset
    :param x: numpy array [N,M]
    :param y: numpy array [N]
    :return: TensorDataset(x,y)
    """
    # use as tensor to avoid array copy
    x = torch.as_tensor(x).float().to(device)
    y = torch.as_tensor(y).long().to(device)
    return TensorDataset(x, y)


def shuffle_data(data: [np.ndarray, np.ndarray]) -> [np.ndarray, np.ndarray]:
    """
    Shuffle a couple of numpy arrays.
    :param data:  arrays of same size ([N,M] [N,O]).
    :return: shuffled data.
    """
    dsize = data[0].shape[0]  # get data size
    indexes = np.random.permutation(dsize)  # shuffle indexes
    # reindex the data
    return [data[0][indexes], data[1][indexes]]


def _random_split(labels: np.ndarray, ratio: Union[float, int] = 0.2) -> [np.ndarray, np.ndarray]:
    """
    Randomly split a dataset based on its labels.
    :param labels: dataset labels.
    :param ratio: validation size or percentage.
    :return: [train_index, validation_index]
    """
    dsize = len(labels)  # get data size
    # ratio to int size
    split_size = int(ratio) if ratio > 1 else int(dsize * ratio)

    # shuffle and split indexes
    indexes = np.random.permutation(dsize)
    train_idx, test_idx = indexes[:-split_size], indexes[-split_size:]

    # return splited data
    return train_idx, test_idx


def _random_ratio_split(labels: np.ndarray, ratio: Union[float, int] = 0.2) -> [np.ndarray, np.ndarray]:
    """
    Randomly split a dataset while maintaining class distribution.
    :param labels: dataset labels.
    :param ratio: validation size or percentage.
    :return: [train_index, validation_index]
    """
    dsize = len(labels)  # get data size
    # get each class size and compute split sizes
    classes, counts = np.unique(labels, return_counts=True)
    ratio = int(ratio) if ratio > 1 else int(ratio * dsize)
    sizes = ratio * counts / dsize  # split sizes
    sizes = sizes.astype(int)
    # if the sizes don't add up to full size add remaining to smallest
    min_index = sizes.argmin()
    sizes[min_index] += ratio - sum(sizes)

    # first and second part of the data
    train_idx, test_idx = [], []
    indices = np.arange(dsize)

    # split each class data
    for class_id, split_size, class_count in zip(classes, sizes, counts):
        class_indices = indices[labels == class_id]
        # random split data
        train_id, test_id = _random_split(class_indices, ratio=split_size)
        # append to parts
        train_idx.append(train_id)
        test_idx.append(test_id)

    # concatenate the result
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    # return shuffled data
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


def random_split(labels:  np.ndarray, ratio: Union[float, int] = 0.2,
                 keep_ratio: bool = False) -> [np.ndarray, np.ndarray]:
    """
    Randomly split a dataset while conserving class repartition or not.
    :param labels: dataset labels.
    :param ratio: validation size or percentage.
    :param keep_ratio: if we want to conserve class distribution.
    :return: [train_index, validation_index]
    """
    # if we don't want to force ratio conservation
    if not keep_ratio:
        train_idx, test_idx = _random_split(labels, ratio=ratio)
    # else force ratio conservation
    else:
        train_idx, test_idx = _random_ratio_split(labels, ratio=ratio)

    return train_idx, test_idx


def random_split_data(data, ratio=0.2):
    first_index, second_index = random_split(data[1], ratio=ratio)
    first_data = [data[0][first_index], data[1][first_index]]
    second_data = [data[0][second_index], data[1][second_index]]
    return first_data, second_data
