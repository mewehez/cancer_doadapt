import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from dnadapt.data.dataset import CustomDataset
from dnadapt.globals import device
from dnadapt.utils.progressBar import make_progressbar
from dnadapt.summary.watcher import StatsData
from dnadapt.utils.functions import accuracy, init_weights


def _make_loader(data, bsize=32):
    src_size, trg_size = data[0].shape[0], data[1].shape[0]
    y = np.concatenate([np.zeros(src_size), np.ones(trg_size)])
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    x = np.concatenate(data)
    dataset = CustomDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True)  # NOTE: Always shuffle
    return dataloader


def _run_epoch(model, loader, loss_fn, opt=None, train=True, watcher=None, **kwargs):
    if train and opt is None:
        raise RuntimeError('Optimizer is required in training mode.')
    # define some metrics
    if watcher is None:
        watcher = StatsData()

    # progress bar
    loader_size = len(loader)  # number of batches
    name = 'DTra' if train else 'DEva'
    pbar = make_progressbar(loader_size, name=name)

    for i, data in enumerate(loader, 1):
        x, y = data
        z = model(x)
        loss = loss_fn(z, y)
        acc = accuracy(z, y)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # append metrics
        watcher.update({'loss': loss.item(), 'acc': acc})

        # step progress bar
        stats = {
            'loss': '%.4f' % np.mean(watcher.data_run['loss']),
            'acc': '%.4f' % np.mean(watcher.data_run['acc'])
        }
        pbar(step=1, vals=stats)
        del loss
    watcher.snap()


def _make_training(model, loader, loss_fn, opt):
    def train(watcher=None, **kwargs):
        model.train()
        return _run_epoch(model, loader, loss_fn, opt=opt, watcher=watcher, **kwargs)
    return train


def _make_validation(model, loader, loss_fn):
    @torch.no_grad()
    def validate(watcher=None, **kwargs):
        model.eval()
        return _run_epoch(model, loader, loss_fn, train=False, watcher=watcher, **kwargs)
    return validate


class Discriminator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optim.Adam(self.model.parameters(), lr=1e-4)

    def reset(self):
        # optimizer.param_groups[0]['lr']
        init_weights(self.model)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-4)

    def __call__(self, train_data, valid_data=None, bsize=32, epochs=20):
        train_loader = _make_loader(train_data, bsize=bsize)
        train_stats = StatsData()
        training_fn = _make_training(self.model, train_loader, self.loss_fn, self.opt)

        # validation
        validation_fn, valid_stats = None, None
        if valid_data is not None:
            valid_stats = StatsData()
            valid_loader = _make_loader(valid_data, bsize=bsize)
            validation_fn = _make_validation(self.model, valid_loader, self.loss_fn)

        print("\nTraining discriminator:")
        for i in range(1, epochs + 1):
            print("[Epoch  {:2d}/{:2d}]".format(i, epochs))
            # train and validate epoch
            training_fn(watcher=train_stats)
            if validation_fn is not None:
                validation_fn(watcher=valid_stats)
        print('')

        # get optimal values
        idx = np.argmin(train_stats.data['loss'])
        stats = {
            'train_loss': train_stats.data['loss'][idx], 'train_acc': train_stats.data['acc'][idx],
        }
        if valid_data is not None:
            stats['valid_loss'] = valid_stats.data['loss'][idx]
            stats['valid_acc'] = valid_stats.data['acc'][idx]
        return stats
