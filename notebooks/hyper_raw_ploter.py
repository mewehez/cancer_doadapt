import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
sns.set(style="whitegrid", palette="muted", color_codes=True)


def disc_raw_plot(path, meta, figsize=None, index_col=0, param_name='param'):
    ncols = meta.shape[0]
    if figsize is None:
        figsize = (7*ncols, 7)
        
    f, axes = plt.subplots(nrows=2, ncols=ncols, figsize=figsize, sharex='col')
    for (i, conf) in enumerate(meta.values):
        # df = pd.read_csv(os.path.join(path, conf[1], 'disc'), index_col=index_col)
        df = np.load(os.path.join(path, conf[1], 'disc' + '.npz'), allow_pickle=True)
        df = df['arr_0'].item()
        # sns.lineplot(data=df[['train_acc', 'valid_acc']], ax=axes[0, i], dashes=False)
        axes[0, i].plot(df['train_acc'], label='Training', marker='.')
        axes[0, i].plot(df['valid_acc'], label='Validation', marker='x')
        axes[0, i].set_title(f'{param_name}={conf[0]}')

        # sns.lineplot(data=df[['train_loss', 'valid_loss']], ax=axes[1, i], dashes=False)
        axes[1, i].plot(df['train_loss'], label='Training', marker='.')
        axes[1, i].plot(df['valid_loss'], label='Validation', marker='x')
        axes[1, i].set_xlabel('Epoch')

    axes[0, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylabel('Loss')
    f.suptitle('Domain discriminator')
    plt.show()


def src_trg_raw_plot(path, file, meta, figsize=None, index_col=0, param_name=''):
    """Plot source and target losses and accuracies."""
    ncols = meta.shape[0]
    if figsize is None:
        figsize = (7*ncols, 7)
        
    f, axes = plt.subplots(nrows=2, ncols=ncols, figsize=figsize, sharex='col')
    for (i, conf) in enumerate(meta.values):
        # df = pd.read_csv(os.path.join(path, conf[1], file), index_col=index_col)
        df = np.load(os.path.join(path, conf[1], file + '.npz'), allow_pickle=True)
        df = df['arr_0'].item()

        """Plot source and target losses and accuracies."""
        axes[0, i].plot(df['ac_s'], label='source')
        axes[0, i].plot(df['ac_t'], label='target')
        axes[0, i].set_title(f'{param_name}={conf[0]}')
        axes[0, i].legend()

        axes[1, i].plot(df['lc_s1'], label='source')
        axes[1, i].plot(df['lc_t'], label='target')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].legend()

    axes[0, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylabel('Loss')
    f.suptitle('Source vs Target')
    plt.show()

    
def wasserstein_raw_plot(path, file, meta, figsize=None, index_col=0, gr_ylim=None, wd_ylim=None, param_name=''):
    """Plot Wassestein distance and gradient penalization."""
    ncols = meta.shape[0]
    if figsize is None:
        figsize = (7*ncols, 7)
    
    f, axes = plt.subplots(nrows=2, ncols=ncols, figsize=figsize, sharex='col')
    for (i, conf) in enumerate(meta.values):
        # df = pd.read_csv(os.path.join(path, conf[1], file), index_col=index_col)
        df = np.load(os.path.join(path, conf[1], file + '.npz'), allow_pickle=True)
        df = df['arr_0'].item()

        # gradient
        axes[0, i].plot(df['wd_grad'])
        axes[0, i].set_ylim(gr_ylim)
        axes[0, i].set_title(f'{param_name}={conf[0]}')

        # wassestein loss
        axes[1, i].plot(-np.array(df['ewd_loss']), label='estimated')
        axes[1, i].plot(df['lwd'], label='computed')
        axes[1, i].set_ylim(wd_ylim)
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].legend()

    axes[0, 0].set_ylabel('Gradient')
    axes[1, 0].set_ylabel('Loss')
    f.suptitle('Wasserstein')
    plt.show()

    
def wasserstein_valid_raw_plot(path, file, meta, figsize=None, index_col=0, wd_ylim=None, param_name=''):
    ncols = meta.shape[0]
    if figsize is None:
        figsize = (7*ncols, 4)
    
    f, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharex='col', sharey='row')
    for (i, conf) in enumerate(meta.values):
        # df = pd.read_csv(os.path.join(path, conf[1], file), index_col=index_col)
        df = np.load(os.path.join(path, conf[1], file + '.npz'), allow_pickle=True)
        df = df['arr_0'].item()
        # wassestein loss
        axes[i].plot(df['lwd'])
        axes[i].set_ylim(wd_ylim)
        axes[i].set_xlabel('Epoch')
        axes[i].set_title(f'{param_name}={conf[0]}')
    axes[0].set_ylabel('Loss')
    f.suptitle('Wasserstein')
    plt.plot()

    
def losses_raw_plot(path, file, meta, figsize=None, index_col=0, param_name=''):
    """Plot total loss, l2 loss and source classification loss."""
    ncols = meta.shape[0]
    if figsize is None:
        figsize = (7*ncols, 7)
    
    f, axes = plt.subplots(nrows=2, ncols=ncols, figsize=figsize, sharex='col')
    for (i, conf) in enumerate(meta.values):
        # df = pd.read_csv(os.path.join(path, conf[1], file), index_col=index_col)
        df = np.load(os.path.join(path, conf[1], file + '.npz'), allow_pickle=True)
        df = df['arr_0'].item()

        #axes[0, i].plot(df['l2_loss'])
        #axes[0, i].set_title(f'{param_name}={conf[0]}')
        
        sc_loss = 'lc_s' if file == 'valid' else 'lc_t'
        axes[1, i].plot(df['loss'], label='total')
        axes[1, i].plot(df[sc_loss], label='source classif')

        axes[1, i].set_xlabel('Epoch')
        axes[1, i].legend()

    axes[0, 0].set_ylabel('L2 norm')
    axes[1, 0].set_ylabel('Loss')
    f.suptitle('Losses')
    plt.show()
