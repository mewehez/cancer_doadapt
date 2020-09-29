import numpy as np
import torch
from torch import nn, optim
from torch.autograd import grad

from dnadapt.data.wdgrlLoader import make_wdgrl_loader, get_gen_data
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.summary.watcher import StatsData
from dnadapt.utils.data import as_tensor_dataset
from dnadapt.utils.earlyStoping import EarlyStopping
from dnadapt.utils.functions import l2_loss, accuracy, correct_pred
from dnadapt.globals import device
from dnadapt.utils.progressBar import make_progressbar


def data_to_set(data):
    src_set = as_tensor_dataset(*(data[0]))
    trg_set = as_tensor_dataset(*(data[1]))
    return src_set, trg_set


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
    h_hat = torch.cat([h, hs, ht], dim=0).requires_grad_()
    # h_hat = h.requires_grad_()

    fw_h_hat = model(h_hat)
    dfw_h_hat = grad(fw_h_hat, h_hat,
                     grad_outputs=torch.ones_like(fw_h_hat),
                     retain_graph=True, create_graph=True)[0]
    norm_dfw_h_hat = dfw_h_hat.norm(2, dim=1)
    grad_penalty = ((norm_dfw_h_hat - 1) ** 2).mean()
    return grad_penalty


def opt_wd_dist(model, data, optimizer, gamma=10, steps=10):
    hs, ht = data
    ewd_loss, wd_grad = 0.0, 0.0
    # wasserstein distance estimation
    for _ in range(steps):
        lgrad = gradient_penalty(model, [hs, ht])
        # wasserstein loss
        fw_hs, fw_ht = model(hs), model(ht)
        lwd = torch.mean(fw_hs) - torch.mean(fw_ht)

        lwd_star = -lwd + gamma * lgrad  # compute loss
        # optimization step
        optimizer.zero_grad()
        lwd_star.backward()
        optimizer.step()

        # add values to compute mean
        ewd_loss += lwd_star.item()
        wd_grad += lgrad.item()

    return {'ewd_loss': ewd_loss / steps, 'wd_grad': wd_grad / steps}


def _run_epoch(model, loader, clf_loss_fn, train=True, opt=None, wd_opt=None, steps=10,
               wd_param=1, gamma=10, l2_param=1e-5, watcher=None, **kwargs):
    # optimizer is required in training mode
    if train and (opt is None or wd_opt is None):
        raise RuntimeError('Optimizer is required in training mode.')

    # create progressbar
    name = 'Train' if train else 'Valid'
    pbar = make_progressbar(len(loader), name=name)

    # create data stats
    if watcher is None:
        watcher = StatsData()

    # loop over data
    for i, data in enumerate(loader, 1):
        # get the data
        xs, ys = data[0]
        xt, yt = data[1]

        hs, ht = model.gen(xs), model.gen(xt, src=False)

        if train:
            # detach tensor from computational graph and use it as input
            wd_stats = opt_wd_dist(model.w, [hs.detach(), ht.detach()], wd_opt, gamma=gamma, steps=steps)
            watcher.update(wd_stats)

        # train classifier
        zs = model.c(hs)
        # compute losses
        sc_loss = clf_loss_fn(zs, ys)
        lwd = model.w(hs).mean() - model.w(ht).mean()  # wasserstein loss
        params = [param for name, param in model.named_parameters() if 'weight' not in name]
        l2loss = sum([l2_loss(v) for v in params])
        loss = sc_loss + wd_param * lwd + l2_param * l2loss

        # optimization step in training mode
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # compute target loss
        tc_loss, tc_acc = target_loss_acc(model.c, clf_loss_fn, ht, yt)

        # update stats
        watcher.update({
            'l2_loss': l2loss.item(), 'wd_loss': lwd.item(), 'sc_loss': sc_loss.item(), 'total_loss': loss.item(),
            'tc_loss': tc_loss, 'sc_acc': accuracy(zs, ys), 'tc_acc': tc_acc,
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
        model.train(),  # set models to training mode
        return _run_epoch(model, loader, clf_loss_fn, opt=opt, wd_opt=wd_opt, watcher=watcher, **kwargs)
    return train


def make_validation(model, loader, clf_loss_fn):
    @torch.no_grad()
    def validate(watcher=None, **kwargs):
        model.eval()  # set models to evaluation mode
        return _run_epoch(model, loader, clf_loss_fn, train=False, watcher=watcher, **kwargs)
    return validate


def make_model_tester(model):
    @torch.no_grad()
    def test(loader, src=True):
        running_correct, total = 0.0, 0
        accs = []
        loader_size = len(loader)  # number of batches
        pbar = make_progressbar(loader_size, name='Test')  # create progressbar
        model.eval()  # set model to evaluation mode
        for data in loader:
            x, y = data
            h = model.gen(x, src=src)
            z = model.classif(h)  # model output
            # number of correct predictions
            correct = correct_pred(z, y)
            # update running correct and total samples
            running_correct += correct
            total += y.size(0)
            accs.append(correct / y.size(0))  # append batch accuracy
            pbar(vals={'acc': np.mean(accs)})  # step the progressbar

        acc = running_correct / total
        return acc, np.std(accs)  # return accuracy and std
    return test


def train_model(model: WDGRLNet, train_data, valid_data=None, disc=None, epochs=30, lr=1e-3, lr_wd=1e-3, bsize=32,
                patience=3, **kwargs):
    # define optimizers and loss function
    wd_opt = optim.Adam(model.critic_params(), lr=lr_wd)  # wd loss optimizer
    opt = optim.Adam(model.classif_params() + model.gen_params(), lr=lr)  # total loss optimizer
    clf_loss_fn = nn.CrossEntropyLoss()  # classifier loss function

    # Prepare training
    src_trainset, trg_trainset = data_to_set(train_data)
    train_loader = make_wdgrl_loader([src_trainset, trg_trainset], bsize=bsize)
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
        print("[Epoch  {:2d}/{:2d}]".format(i+1, epochs))
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
