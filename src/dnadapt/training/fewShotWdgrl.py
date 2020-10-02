import numpy as np
from torch import nn, optim

from dnadapt.data.dataset import CustomDataset
from dnadapt.data.fewShotWgrlLoader import make_fs_wdgrl_loader
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.training.wdgrl import opt_wd_dist, target_loss_acc, data_to_set, optimize_fnc, \
    create_valid_fnc, train_wdgrl, data_to_custom_set
from dnadapt.utils.data import as_tensor_dataset
from dnadapt.utils.progressBar import make_progressbar
from dnadapt.summary.watcher import StatsData
from dnadapt.utils.functions import accuracy


def _run_train_epoch(model, loader, lc_fnc, g_opt, c_opt, wd_opt, steps=10, lambd=1, gamma=10, watcher=None, **kwargs):
    model.train()
    # create data stats
    if watcher is None: watcher = StatsData()
    # create progress bar
    pbar = make_progressbar(len(loader), name='Train')

    # loop over data
    for i, data in enumerate(loader, 1):
        # get source, target and labeled target data
        [xs, ys], [xt, yt], [xf, yf] = data
        # feature extraction
        hs, ht, hf = model.gen(xs), model.gen(xt, src=False), model.gen(xf, src=False)

        # compute max[lwd - gama * lgrad]
        # ATTENTION - detach tensor from computational graph
        wd_stats = opt_wd_dist(model.w, [hs.detach(), ht.detach()], wd_opt, gamma=gamma, steps=steps)
        watcher.update(wd_stats)

        # train classifier
        sum_lc_s1, sum_lc_f1 = 0.0, 0.0
        for j in range(steps):
            zs, zf = model.c(hs.detach()), model.c(hf.detach())
            lc_s1, lc_f1 = lc_fnc(zs, ys), lc_fnc(zf, yf)
            optimize_fnc(lc_s1+lc_f1, c_opt)
            sum_lc_s1 += lc_s1.item()
            sum_lc_f1 += lc_f1.item()
        lc_s1, lc_f1 = sum_lc_s1/steps, sum_lc_f1/steps

        # train classifier and domain critic
        zs, zf = model.c(hs), model.c(hf)
        # compute loss
        lc_s2, lc_f2 = lc_fnc(zs, ys), lc_fnc(zf, yf)
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        # params = [param for name, param in model.named_parameters() if 'weight' not in name]
        # l2loss = sum([l2_loss(v) for v in params])
        loss = lc_s2 + lc_f2 + lambd * lwd
        optimize_fnc(loss, g_opt)  # optimization step

        # compute accuracies
        lc_t, ac_t = target_loss_acc(model.c, lc_fnc, ht, yt)
        ac_s, ac_f = accuracy(zs, ys), accuracy(zf, yf)

        # update stats
        watcher.update({
            'lwd': lwd.item(), 'lc_s1': lc_s1, 'lc_f1': lc_f1, 'lc_s2': lc_s2.item(), 'lc_f2': lc_f2.item(),
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


def make_training(model, loader, lc_fnc, g_opt, c_opt, w_opt):
    def train(watcher=None, **kwargs):
        return _run_train_epoch(model, loader, lc_fnc, g_opt, c_opt, w_opt, watcher=watcher, **kwargs)

    return train


def train_model(model: WDGRLNet, train_data, valid_data=None, disc=None, epochs=30, alpha1=1e-3, alpha2=1e-3, bsize=32,
                patience=3, min_epoch=10, **kwargs):
    # define optimizers and loss function
    w_opt = optim.Adam(model.w_params(), lr=alpha1)
    g_opt = optim.Adam(model.g_params(), lr=alpha2)
    c_opt = optim.Adam(model.c_params(), lr=alpha2)
    lc_fnc = nn.CrossEntropyLoss()  # classifier loss function

    # Prepare training
    trainset = data_to_custom_set(train_data[:-1])  # train data 0, 1
    trg_fs_set = CustomDataset(*train_data[-1])
    train_loader = make_fs_wdgrl_loader(trainset, trg_fs_set, bsize=bsize)
    training_fnc = make_training(model, train_loader, lc_fnc, g_opt, c_opt, w_opt)

    # Prepare validation
    validation_fnc, validset = create_valid_fnc(model, lc_fnc, valid_data=valid_data, bsize=bsize)
    return train_wdgrl(model, trainset, training_fnc, validset=validset, valid_fnc=validation_fnc, disc=disc,
                       epochs=epochs, patience=patience, min_epoch=min_epoch, **kwargs)
