import os
from torch.utils.data import DataLoader

from dnadapt.data.wdgrlLoader import batch_generator, load_data, make_wdgrl_loader
from dnadapt.utils.data import random_split_data, as_tensor_dataset
from dnadapt.globals import datadir


def make_fs_wdgrl_loader(dataset, fs_dataset, bsize=32):
    src_gen = batch_generator(DataLoader(dataset[0], batch_size=bsize, shuffle=True, drop_last=True))
    trg_gen = batch_generator(DataLoader(dataset[1], batch_size=bsize, shuffle=True, drop_last=True))
    fs_gen = batch_generator(DataLoader(fs_dataset, batch_size=bsize))

    # epoch size
    src_epoch_size = int(len(dataset[0].tensors[1]) / bsize)
    trg_epoch_size = int(len(dataset[1].tensors[1]) / bsize)
    epoch_size = max(src_epoch_size, trg_epoch_size)
    dataloader = FSLoader(src_gen, trg_gen, fs_gen, size=epoch_size)
    return dataloader


class FSLoader:
    def __init__(self, src_gen, trg_gen, fs_gen, size=0):
        self.index = 0
        self.size = size
        self.src_gen = src_gen
        self.trg_gen = trg_gen
        self.fs_gen = fs_gen

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.size

    def __next__(self):
        if self.index < self.size:
            s_batch = next(self.src_gen)
            t_batch = next(self.trg_gen)
            f_batch = next(self.fs_gen)
            self.index += 1
            return [s_batch, t_batch, f_batch]
        else:
            raise StopIteration


if __name__ == '__main__':
    src_path = os.path.join(datadir, 'toy', 'src', 'train.npz')
    trg_path = os.path.join(datadir, 'toy', 'trg', 'train.npz')

    src_train, src_valid = load_data(src_path, valid_size=0.2)
    src_trainset, src_validset = as_tensor_dataset(*src_train), as_tensor_dataset(*src_valid)

    trg_train, trg_valid = load_data(trg_path, valid_size=0.2)
    trg_train, trg_fs = random_split_data(trg_train, ratio=20)
    trg_trainset, trg_validset = as_tensor_dataset(*trg_train), as_tensor_dataset(*trg_valid)
    trg_fs_set = as_tensor_dataset(*trg_fs)

    train_loader = make_fs_wdgrl_loader([src_trainset, trg_trainset], trg_fs_set, bsize=32)
    valid_loader = make_wdgrl_loader([src_validset, trg_validset], bsize=32)

    for i, dt in enumerate(train_loader):
        if i >= 13:
            break
        src, trg, fs = dt
        if i == 10:
            print(dt)
