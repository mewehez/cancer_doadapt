import numpy as np

import torch
from torch.utils.data import DataLoader

from dnadapt.utils.data import random_split_data, as_tensor_dataset


def load_data(data_path, valid_size=None):
    # get training data
    data = np.load(data_path)
    data = [data['x'], data['y']]
    print(f'Loaded data size: {data[0].shape}')  # NOTE:

    if valid_size is None:
        return data, None

    return random_split_data(data, ratio=valid_size)


def create_test_data_loader(src_path, trg_path, bsize=32):
    # load np data
    src_data, _ = load_data(src_path)
    trg_data, _ = load_data(trg_path)

    # create test data loader
    src_set = as_tensor_dataset(*src_data)
    trg_set = as_tensor_dataset(*trg_data)
    src_loader = DataLoader(src_set, batch_size=bsize)
    trg_loader = DataLoader(trg_set, batch_size=bsize)
    return src_loader, trg_loader


def batch_generator(dataloader: DataLoader):
    """
    Creates infinite iterator from DataLoader.
    :param dataloader: data loader from which data is sampled.
    :return: yield data from dataloader.
    """
    while True:
        for data in iter(dataloader):
            yield data


def make_wdgrl_loader(dataset, bsize=32):
    src_gen = batch_generator(DataLoader(dataset[0], batch_size=bsize, shuffle=True, drop_last=True))
    trg_gen = batch_generator(DataLoader(dataset[1], batch_size=bsize, shuffle=True, drop_last=True))

    # compute epoch size
    src_epoch_size = int(len(dataset[0].tensors[1])/bsize)
    trg_epoch_size = int(len(dataset[1].tensors[1])/bsize)
    epoch_size = max(src_epoch_size, trg_epoch_size)
    dataloader = WDLoader(src_gen, trg_gen, size=epoch_size)
    return dataloader


@torch.no_grad()
def get_gen_data(model, dataset):
    xs, _ = dataset[0].tensors
    xt, _ = dataset[1].tensors
    hs, ht = model.gen(xs), model.gen(xt, src=False)
    return hs, ht


class WDLoader:
    """Iterator from two data_generators"""
    def __init__(self, src_generator, trg_generator, size=0):
        self.index = 0
        self.size = size
        self.src_gen = src_generator
        self.trg_gen = trg_generator

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        if self.index < self.size:
            s_batch = next(self.src_gen)
            t_batch = next(self.trg_gen)
            self.index += 1
            return [s_batch, t_batch]
        else:
            raise StopIteration
