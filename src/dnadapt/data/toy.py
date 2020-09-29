import os
import numpy as np
from sklearn.datasets import make_blobs
from dnadapt.utils.data import random_split
from dnadapt.utils.utils import folder_if_not_exist, pcts_to_sizes
from dnadapt.globals import datadir


def blob_data(size=1000, pcts=None, centers=None, std=1.0):
    if pcts is not None:
        size = pcts_to_sizes(size, pcts)
    return make_blobs(n_samples=size, centers=centers, cluster_std=std)


def create_data(src_centers=None, trg_centers=None, src_pcts=None, trg_pcts=None, src_size=1000,
                trg_size=1000, ratio=0.2):
    # create data
    src_data = blob_data(size=src_size, pcts=src_pcts, centers=src_centers, std=1)
    trg_data = blob_data(size=trg_size, pcts=trg_pcts, centers=trg_centers, std=1)
    # split src data
    train_idx, valid_idx = random_split(src_data[1], ratio=ratio)
    src_train = [d[train_idx] for d in src_data]
    src_valid = [d[valid_idx] for d in src_data]
    # split target data
    train_idx, valid_idx = random_split(trg_data[1], ratio=ratio)
    trg_train = [d[train_idx] for d in trg_data]
    trg_valid = [d[valid_idx] for d in trg_data]
    return [src_train, trg_train], [src_valid, trg_valid]


def create_data_file(path, src_centers=None, trg_centers=None, src_pcts=None, trg_pcts=None, src_size=1000,
                     trg_size=1000, ratio=0.2):
    # create data
    [src_train, src_test], [trg_train, trg_test] = create_data(
        src_centers=src_centers, trg_centers=trg_centers, src_pcts=src_pcts, trg_pcts=trg_pcts,
        src_size=src_size, trg_size=trg_size, ratio=ratio)

    src_folder = folder_if_not_exist(os.path.join(path, 'src'))
    np.savez_compressed(os.path.join(src_folder, 'train'), x=src_train[0], y=src_train[1])
    np.savez_compressed(os.path.join(src_folder, 'test'), x=src_test[0], y=src_test[1])

    trg_folder = folder_if_not_exist(os.path.join(path, 'trg'))
    np.savez_compressed(os.path.join(trg_folder, 'train'), x=trg_train[0], y=trg_train[1])
    np.savez_compressed(os.path.join(trg_folder, 'test'), x=trg_test[0], y=trg_test[1])


if __name__ == '__main__':
    data_config = {
        'src_size': 5000,
        'trg_size': 1000,
        'src_pcts': [0.37, 0.63],
        'trg_pcts': [0.37, 0.63],
        'src_centers': [[0, 0], [0, 10]],
        'trg_centers': [[50, -20], [50, -10]]
    }

    create_data_file(os.path.join(datadir, 'toy'), **data_config)
