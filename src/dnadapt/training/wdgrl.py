import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad

from dnadapt.data.dataset import CustomDataset
from dnadapt.data.wdgrlLoader import make_wdgrl_loader, get_gen_data
from dnadapt.models.discriminator import Discriminator
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.summary.watcher import StatsData, DataWatcher
from dnadapt.utils.data import as_tensor_dataset
from dnadapt.utils.earlyStoping import EarlyStopping
from dnadapt.utils.functions import accuracy, correct_pred
from dnadapt.globals import device
from dnadapt.utils.progressBar import make_progressbar


def data_to_custom_set(data):
    src_set = CustomDataset(*(data[0]))
    trg_set = CustomDataset(*(data[1]))
    return src_set, trg_set


def data_to_set(data):
    src_set = as_tensor_dataset(*(data[0]))
    trg_set = as_tensor_dataset(*(data[1]))
    return src_set, trg_set


def create_valid_fnc(model, lc_fnc, valid_data=None, bsize=32):
    valid_fnc, validset = None, None
    if valid_data is not None:
        validset = data_to_custom_set(valid_data)
        valid_loader = make_wdgrl_loader(validset, bsize=bsize)
        valid_fnc = make_validation(model, valid_loader, lc_fnc)
    return valid_fnc, validset


def optimize_fnc(fnc, optimizer):
    optimizer.zero_grad()
    fnc.backward()
    optimizer.step()


@torch.no_grad()
def target_loss_acc(model, loss_fn, h, labels) -> [float, float]:
    z = model(h)
    loss = loss_fn(z, labels)
    acc = accuracy(z, labels)
    return loss.item(), acc


def gradient_penalty(model, data):
    hs, ht = data
    alpha = torch.rand(hs.size(0), 1, device=device)
    h = alpha * hs + (1.0 - alpha) * ht  # random point on the segment [hs, ht]
    h_hat = h.requires_grad_()
    # h_hat = torch.cat([h, hs, ht], dim=1).requires_grad_()  # stack variables

    fw_h_hat = model(h_hat)
    dfw_h_hat = grad(fw_h_hat, h_hat,
                     grad_outputs=torch.ones_like(fw_h_hat),
                     retain_graph=True, create_graph=True)[0]
    norm_dfw_h_hat = dfw_h_hat.norm(2, dim=1)
    grad_penalty = ((norm_dfw_h_hat - 1) ** 2).mean()
    return grad_penalty


def opt_wd_dist(model, data, optimizer, gamma=10, steps=10):
    hs, ht = data
    sum_lwd_, sum_lgrad = 0.0, 0.0
    # wasserstein distance estimation
    for _ in range(steps):
        lgrad = gradient_penalty(model, [hs, ht])
        # wasserstein loss
        fw_hs, fw_ht = model(hs), model(ht)
        lwd = torch.mean(fw_hs) - torch.mean(fw_ht)

        lwd_ = -lwd + gamma * lgrad  # compute loss
        optimize_fnc(lwd_, optimizer)  # optimization step

        # add values to compute mean
        sum_lwd_ += lwd_.item()
        sum_lgrad += lgrad.item()

    return {'lwd_': sum_lwd_ / steps, 'lgrad': sum_lgrad / steps}


def _run_train_epoch(model, loader, lc_fnc, opts, steps=10, lambd=1, gamma=10, watcher=None, **kwargs):
    model.train()
    # create watcher if none
    if watcher is None: watcher = StatsData()
    # create progress bar
    pbar = make_progressbar(len(loader), name='Train')

    # loop over data
    for i, data in enumerate(loader, 1):
        [xs, ys], [xt, yt] = data  # get src and trg data
        hs, ht = model.gen(xs), model.gen(xt, src=False)  # feature extraction

        # compute max[lwd - gama * lgrad]
        # ATTENTION - detach tensor from computational graph
        wd_stats = opt_wd_dist(model.w, [hs.detach(), ht.detach()], opts['w_opt'], gamma=gamma, steps=steps)
        watcher.update(wd_stats)

        # train classifier
        zs = model.c(hs)
        lc_s = lc_fnc(zs, ys)
        optimize_fnc(lc_s, opts['sc_opt'])
        ac_s = accuracy(zs, ys)  # source classification accuracy

        # train domain critic
        hs, ht = model.gen(xs), model.gen(xt, src=False)  # feature extraction
        # zs = model.c(hs)
        # lc_s2 = lc_fnc(zs, ys)  # compute lc_s again since weights have back propagated
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        # params = [param for name, param in model.named_parameters() if 'weight' not in name]
        # l2loss = sum([l2_loss(v) for v in params])
        # loss = lwd
        optimize_fnc(lwd, opts['t_opt'])  # optimization step

        # target classification loss and accuracy
        lc_t, ac_t = target_loss_acc(model.c, lc_fnc, ht, yt)

        # update stats
        watcher.update({
            'lwd': lwd.item(), 'lc_s1': lc_s.item(), 'loss': lwd.item(),
            'lc_t': lc_t, 'ac_s': ac_s, 'ac_t': ac_t,
        })

        # update progressbar
        bar_values = {
            'trg': '%.4f' % np.mean(watcher.data_run['ac_t']),
            'src': '%.4f' % np.mean(watcher.data_run['ac_s'])
        }
        pbar(step=1, vals=bar_values)

    watcher.snap()
    return watcher


@torch.no_grad()
def _run_valid_epoch(model, loader, lc_fnc, lambd=1, watcher=None, **kwargs):
    model.eval()
    # create watcher if none
    if watcher is None: watcher = StatsData()

    # create progress bar
    pbar = make_progressbar(len(loader), name='Valid')

    # loop over data
    for i, data in enumerate(loader, 1):
        [xs, ys], [xt, yt] = data  # get src and trg data
        hs, ht = model.gen(xs), model.gen(xt, src=False)  # feature extraction

        # compute losses
        zs, zt = model.c(hs), model.c(ht)
        lc_s, lc_t = lc_fnc(zs, ys), lc_fnc(zt, yt)
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        # all_variables = [param for name, param in model.named_parameters() if 'weight' not in name]
        # l2loss = sum([l2_loss(v) for v in all_variables])
        loss = lc_s + lambd * lwd

        # compute accuracies
        ac_s, ac_t = accuracy(zs, ys), accuracy(zt, yt)
        # update stats
        watcher.update({
            'lwd': lwd.item(), 'lc_s': lc_s.item(), 'loss': loss.item(),
            'lc_t': lc_t.item(), 'ac_s': ac_s, 'ac_t': ac_t,
        })

        # update progressbar
        bar_values = {
            'trg': '%.4f' % np.mean(watcher.data_run['ac_t']),
            'src': '%.4f' % np.mean(watcher.data_run['ac_s'])
        }
        pbar(step=1, vals=bar_values)

    watcher.snap()
    return watcher


def make_training(model, loader, lc, opts):
    def train(watcher=None, **kwargs):
        return _run_train_epoch(model, loader, lc, opts, watcher=watcher, **kwargs)

    return train


def make_validation(model, loader, lc_fnc):
    def validate(watcher=None, **kwargs):
        return _run_valid_epoch(model, loader, lc_fnc, watcher=watcher, **kwargs)

    return validate


def make_model_tester(model):
    @torch.no_grad()
    def test(loader, src=True):
        running_correct, total, accs = 0.0, 0, []
        loader_size = len(loader)  # number of batches
        pbar = make_progressbar(loader_size, name='Test')  # create progressbar
        model.eval()  # set model to evaluation mode
        for data in loader:
            x, y = data
            h = model.gen(x, src=src)
            z = model.c(h)  # model output
            # number of correct predictions
            correct = correct_pred(z, y)
            # update running correct and total samples
            running_correct += correct
            total += y.size(0)
            accs.append(correct / y.size(0))  # append batch accuracy
            pbar(vals={'acc': np.mean(accs)})  # step the progressbar

        return running_correct / total

    return test


def train_wdgrl(model: WDGRLNet, trainset, train_fnc, validset=None, valid_fnc=None, disc: Discriminator = None,
                epochs=30, patience=3, min_epoch=10, **kwargs):
    # create data watcher
    watcher = DataWatcher()
    watcher.add_data('train')
    if valid_fnc is not None:
        watcher.add_data('valid')
    if disc is not None:
        watcher.add_data('disc')

    stopper = EarlyStopping(patience=patience)  # prepare early stopping
    # training loop
    for i in range(epochs):
        print("[Epoch  {:2d}/{:2d}]".format(i + 1, epochs))
        stats = train_fnc(watcher=watcher.data['train'], **kwargs)
        train_loss, train_acc = stats.data['loss'][-1], stats.data['ac_t'][-1]

        # epoch validation
        if valid_fnc is not None:
            stats = valid_fnc(watcher=watcher.data['valid'], **kwargs)
            valid_loss, valid_acc = stats.data['loss'][-1], stats.data['ac_t'][-1]
            stopper(model, valid_loss, train_acc, valid_acc)
        else:
            stopper(model, train_loss, train_acc, None)

        # discriminator training
        if disc is not None:
            train_data_ = get_gen_data(model, trainset)

            valid_data_ = None
            if validset is not None:
                valid_data_ = get_gen_data(model, validset)
            stats = disc(train_data_, valid_data=valid_data_)
            watcher.add_data(name='disc', data=stats)
            disc.reset()  # reset weights
            del [train_data_, valid_data_]

        if stopper.stop() and i >= min_epoch:
            break
    return watcher


def train_model(model: WDGRLNet, train_data, valid_data=None, disc=None, epochs=30, alpha1=1e-3, alpha2=1e-3, bsize=32,
                patience=3, min_epoch=10, **kwargs):
    # define optimizers and loss function
    w_opt = optim.Adam(model.w_params(), lr=alpha1)
    sc_opt = optim.Adam(model.gs_params() + model.c_params(), lr=alpha2)
    gt_opt = optim.Adam(model.gt_params(), lr=alpha2)
    # c_opt = optim.Adam(model.c_params(), lr=alpha2)
    lc_fnc = nn.CrossEntropyLoss()  # classification loss function

    # Prepare training
    trainset = data_to_custom_set(train_data)
    train_loader = make_wdgrl_loader(trainset, bsize=bsize)
    opts = {'sc_opt': sc_opt, 't_opt': gt_opt, 'w_opt': w_opt}
    training_fnc = make_training(model, train_loader, lc_fnc, opts)

    # Prepare validation
    validation_fnc, validset = create_valid_fnc(model, lc_fnc, valid_data=valid_data, bsize=bsize)
    return train_wdgrl(model, trainset, training_fnc, validset=validset, valid_fnc=validation_fnc, disc=disc,
                       epochs=epochs, patience=patience, min_epoch=min_epoch, **kwargs)
