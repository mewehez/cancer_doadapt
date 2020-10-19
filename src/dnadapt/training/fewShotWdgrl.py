import numpy as np
import torch
from geomloss import SamplesLoss
from torch import nn, optim

from dnadapt.data.dataset import CustomDataset
from dnadapt.data.fewShotWgrlLoader import make_fs_wdgrl_loader
from dnadapt.models.discriminator import Discriminator
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.training.wdgrl import opt_wd_dist, target_loss_acc, data_to_set, optimize_fnc, \
    create_valid_fnc, train_wdgrl, data_to_custom_set
from dnadapt.utils.data import as_tensor_dataset
from dnadapt.utils.progressBar import make_progressbar
from dnadapt.summary.watcher import StatsData
from dnadapt.utils.functions import accuracy, l2_loss


def _run_train_epoch(model, loader, lc_fnc, opts, steps=10, lambd=1, gamma=10, eps=1e-3, watcher=None, **kwargs):
    model.train()
    # create data stats
    if watcher is None: watcher = StatsData()
    # create progress bar
    pbar = make_progressbar(len(loader), name='Train')

    sinkhorn = SamplesLoss("sinkhorn", p=1, blur=.01)

    # loop over data
    for i, data in enumerate(loader, 1):
        # get source, target and labeled target data
        [xs, ys], [xt, yt], [xf, yf] = data
        # feature extraction
        hs, ht, hf = model.gen(xs), model.gen(xt, src=False), model.gen(xf, src=False)

        # compute max[lwd - gama * lgrad]
        # ATTENTION - detach tensor from computational graph
        wd_stats = opt_wd_dist(model.w, [hs.detach(), ht.detach()], opts['w_opt'], gamma=gamma, steps=steps)
        watcher.update(wd_stats)

        # train classifier
        zs = model.c(hs.detach())
        lc_s = lc_fnc(zs, ys)
        optimize_fnc(lc_s, opts['sc_opt'])
        ac_s = accuracy(zs, ys)  # source classification accuracy

        # train classifier and domain critic
        zf = model.c(hf)
        lc_f = lc_fnc(zf, yf)

        # compute loss
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        params = [param for name, param in model.named_parameters() if 'weight' not in name]
        l2loss = sum([l2_loss(v) for v in params])
        loss = lc_f + lambd * lwd + eps * l2loss
        optimize_fnc(loss, opts['t_opt'])  # optimization step

        # compute accuracies
        lc_t, ac_t = target_loss_acc(model.c, lc_fnc, ht, yt)
        ac_f = accuracy(zf, yf)

        # sinkhorn dist
        with torch.no_grad():
            skh_loss = sinkhorn(hs, ht)

        # update stats
        watcher.update({
            'lwd': lwd.item(), 'lc_s': lc_s.item(), 'lc_f': lc_f.item(), 'skh_loss': skh_loss.item(),
            'loss': loss.item(), 'lc_t': lc_t, 'ac_s': ac_s, 'ac_t': ac_t, 'ac_f': ac_f
        })

        # update progressbar
        bar_values = {
            'trg': '%.4f' % np.mean(watcher.data_run['ac_t']),
            'fst': '%.4f' % np.mean(watcher.data_run['ac_f']),
            'src': '%.4f' % np.mean(watcher.data_run['ac_f'])
        }
        pbar(step=1, vals=bar_values)

    watcher.snap()
    return watcher


def make_training(model, loader, clf_loss_fn, opts):
    def train(watcher=None, **kwargs):
        return _run_train_epoch(model, loader, clf_loss_fn, opts, watcher=watcher, **kwargs)
    return train


def train_model(model: WDGRLNet, train_data, valid_data=None, disc: Discriminator = None, epochs=30, alpha1=1e-3,
                alpha2=1e-3, bsize=32, patience=5, min_epoch=10, **kwargs):
    # define optimizers and loss function
    w_opt = optim.Adam(model.w_params(), lr=alpha1)  # wd loss optimizer
    sc_opt = optim.Adam(model.gs_params() + model.c_params(), lr=alpha2)  # total loss optimizer
    t_opt = optim.Adam(model.gt_params(), lr=alpha2)
    lc_fnc = nn.CrossEntropyLoss()  # classifier loss function

    # Prepare training
    opts = {'w_opt': w_opt, 'sc_opt': sc_opt, 't_opt': t_opt}
    trainset = data_to_custom_set(train_data[:-1])  # train data 0, 1
    trg_fs_set = CustomDataset(*train_data[-1])
    train_loader = make_fs_wdgrl_loader(trainset, trg_fs_set, bsize=bsize)
    training_fnc = make_training(model, train_loader, lc_fnc, opts)

    # Prepare validation
    validation_fnc, validset = create_valid_fnc(model, lc_fnc, valid_data=valid_data, bsize=bsize)
    return train_wdgrl(model, trainset, training_fnc, validset=validset, valid_fnc=validation_fnc, disc=disc,
                       epochs=epochs, patience=patience, min_epoch=min_epoch, **kwargs)
