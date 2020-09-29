import numpy as np
import torch
from torch import nn, optim

from dnadapt.data.fewShotWgrlLoader import make_fs_wdgrl_loader
from dnadapt.data.wdgrlLoader import make_wdgrl_loader, get_gen_data
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.training.wdgrl import opt_wd_dist, target_loss_acc, data_to_set
from dnadapt.utils.data import as_tensor_dataset
from dnadapt.utils.progressBar import make_progressbar
from dnadapt.utils.earlyStoping import EarlyStopping
from dnadapt.summary.watcher import StatsData
from dnadapt.utils.functions import l2_loss, accuracy


def _run_train_epoch(model, loader, clf_loss_fn, opt, wd_opt, steps=10, lambd=1, gamma=10,
                     l2_param=1e-5, watcher=None, **kwargs):
    model.train()
    pbar = make_progressbar(len(loader), name='Train')
    # create data stats
    if watcher is None: watcher = StatsData()

    # loop over data
    for i, data in enumerate(loader, 1):
        # get the data
        xs, ys = data[0]
        xt, yt = data[1]
        xf, yf = data[2]

        # feature extraction
        hs, ht = model.gen(xs), model.gen(xt, src=False)
        hf = model.gen(xf, src=False)

        # detach tensor from computational graph and use it as input
        wd_stats = opt_wd_dist(model.w, [hs.detach(), ht.detach()], wd_opt, gamma=gamma, steps=steps)
        watcher.update(wd_stats)

        # train classifier
        zs, zf = model.c(hs), model.c(hf)
        # compute loss
        sc_loss, fc_loss = clf_loss_fn(zs, ys), clf_loss_fn(zf, yf)
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        params = [param for name, param in model.named_parameters() if 'weight' not in name]
        l2loss = sum([l2_loss(v) for v in params])
        loss = sc_loss + fc_loss + lambd * lwd + l2_param*l2loss

        # optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # compute accuracies
        tc_loss, tc_acc = target_loss_acc(model.c, clf_loss_fn, ht, yt)
        sc_acc, fc_acc = accuracy(zs, ys), accuracy(zf, yf)

        # update stats
        watcher.update({
            'l2_loss': l2loss.item(), 'wd_loss': lwd.item(),
            'sc_loss': sc_loss.item(), 'fc_loss': fc_loss.item(),
            'total_loss': loss.item(), 'tc_loss': tc_loss,
            'sc_acc': sc_acc, 'tc_acc': tc_acc, 'fc_acc': fc_acc
        })

        bar_values = {
            'trg': '%.4f' % np.mean(watcher.data_run['tc_acc']),
            'fst': '%.4f' % np.mean(watcher.data_run['fc_acc']),
            'src': '%.4f' % np.mean(watcher.data_run['sc_acc'])
        }
        pbar(step=1, vals=bar_values)
    watcher.snap()
    return watcher


@torch.no_grad()
def _run_valid_epoch(model, loader, clf_loss_fn, lambd=1, l2_param=1e-5, watcher=None, **kwargs):
    model.eval()
    # create progressbar
    pbar = make_progressbar(len(loader), name='Valid')

    # create data stats
    if watcher is None: watcher = StatsData()

    # loop over data
    for i, data in enumerate(loader, 1):
        # get the data
        xs, ys = data[0]
        xt, yt = data[1]

        hs, ht = model.gen(xs), model.gen(xt, src=False)

        # train classifier
        zs, zt = model.c(hs), model.c(ht)
        # compute losses
        sc_loss, tc_loss = clf_loss_fn(zs, ys), clf_loss_fn(zt, yt)
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        all_variables = [param for name, param in model.named_parameters() if 'weight' not in name]
        l2loss = sum([l2_loss(v) for v in all_variables])
        loss = sc_loss + lambd * lwd + l2_param * l2loss

        # compute accuracies
        sc_acc, tc_acc = accuracy(zs, ys), accuracy(zt, yt)

        # update stats
        watcher.update({
            'l2_loss': l2loss.item(), 'wd_loss': lwd.item(), 'sc_loss': sc_loss.item(),
            'total_loss': loss.item(), 'tc_loss': tc_loss,
            'sc_acc': sc_acc, 'tc_acc': tc_acc,
        })

        bar_values = {
            'trg': '%.4f' % np.mean(watcher.data_run['tc_acc']),
            'src': '%.4f' % np.mean(watcher.data_run['sc_acc'])
        }
        pbar(step=1, vals=bar_values)
    watcher.snap()
    return watcher


def make_training(model, loader, clf_loss_fn, opt, wd_opt):
    def train(watcher=None, **kwargs):
        return _run_train_epoch(model, loader, clf_loss_fn, opt, wd_opt, watcher=watcher, **kwargs)
    return train


def make_validation(model, loader, clf_loss_fn):
    def evaluate(watcher=None, **kwargs):
        return _run_valid_epoch(model, loader, clf_loss_fn, watcher=watcher, **kwargs)
    return evaluate


def train_model(model: WDGRLNet, train_data, valid_data=None, disc=None, epochs=30, lr=1e-3, lr_wd=1e-3, bsize=32,
                patience=3, **kwargs):
    # define optimizers and loss function
    wd_opt = optim.Adam(model.critic_params(), lr=lr_wd)  # wd loss optimizer
    opt = optim.Adam(model.classif_params() + model.gen_params(), lr=lr)  # total loss optimizer
    clf_loss_fn = nn.CrossEntropyLoss()  # classifier loss function

    # Prepare training
    src_trainset, trg_trainset = data_to_set(train_data[:-1])
    trg_fs_set = as_tensor_dataset(*train_data[-1])
    train_loader = make_fs_wdgrl_loader([src_trainset, trg_trainset], trg_fs_set, bsize=bsize)
    training_fnc = make_training(model, train_loader, clf_loss_fn, opt, wd_opt)

    # Prepare validation
    validation_fnc, src_validset, trg_validset = None, None, None
    if valid_data is not None:
        src_validset, trg_validset = data_to_set(valid_data)
        valid_loader = make_wdgrl_loader([src_validset, trg_validset], bsize=bsize)
        validation_fnc = make_validation(model, valid_loader, clf_loss_fn)

    stopper = EarlyStopping(patience=patience)  # prepare early stopping
    # training loop
    for i in range(epochs):
        print("[Epoch  {:2d}/{:2d}]".format(i + 1, epochs))
        watcher = training_fnc(**kwargs)
        train_loss, train_acc = watcher.data['total_loss'][0], watcher.data['tc_acc'][0]

        # epoch validation
        if validation_fnc is not None:
            watcher = validation_fnc(**kwargs)
            valid_loss, valid_acc = watcher.data['total_loss'][0], watcher.data['tc_acc'][0]
            stopper(model, valid_loss, train_acc, valid_acc)
        else:
            stopper(model, train_loss, train_acc, None)

        # discriminator training
        if disc is not None:
            disc_train_data = get_gen_data(model, [src_trainset, trg_trainset])

            disc_valid_data = None
            if valid_data is not None:
                disc_valid_data = get_gen_data(model, [src_validset, trg_validset])
            disc(disc_train_data, valid_data=disc_valid_data)
            disc.reset()  # reset weights

        if stopper.stop():
            break
