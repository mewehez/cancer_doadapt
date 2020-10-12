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


def join_data(path, file, meta, index_col=0):
    dt = {}
    index = []
    for (i, conf) in enumerate(meta.values):
        arr = np.load(os.path.join(path, conf[1], file+'.npz'), allow_pickle=True)
        arr = arr['arr_0'].item()
        for key, val in arr.items():
            if key not in dt.keys():
                dt[key] = []
            dt[key].append(val[-1])
        index.append(np.log10(conf[0] + 1e-20))  # add small value to avoid division by zero error
    d = pd.DataFrame(dt, index=index)
    return d


def disc_plot(df, figsize=(14, 4), xlabel=''):
    # prepare subplot
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    axes[0].plot(df['train_acc'], label='Training', marker='.')
    axes[0].plot(df['valid_acc'], label='Validation', marker='x')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    axes[1].plot(df['train_loss'], label='Training', marker='.')
    axes[1].plot(df['valid_loss'], label='Validation', marker='x')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    f.suptitle('Domain discriminator')
    plt.show()


def src_trg_plot(df, figsize=(14, 4), xlabel=''):
    """Plot source and target losses and accuracies."""
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    # sns.lineplot(data=df[['sc_acc', 'tc_acc']], dashes=False, markers=True, ax=axes[0])
    axes[0].plot(df['ac_s'], label='$\mathcal{A}_{C}(S)$', marker='.')
    axes[0].plot(df['ac_t'], label='$\mathcal{A}_{C}(T)$', marker='x')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel(xlabel)
    axes[0].legend()

    # sns.lineplot(data=df[['sc_loss', 'tc_loss']], dashes=False, markers=True, ax=axes[1])
    axes[1].plot(df['lc_s1'], label='$\mathcal{L}_{C}(S)$', marker='.')
    # axes[1].plot(df['lc_s2'], label='$\mathcal{L}_{C}(S2)$', marker='.')
    axes[1].plot(df['lc_t'], label='$\mathcal{L}_{C}(T)$', marker='x')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel(xlabel)
    axes[1].legend()
    
    f.suptitle('Classification')
    plt.show()

    
def src_trg_valid_plot(df, figsize=(14, 4), xlabel=''):
    """Plot source and target losses and accuracies."""
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    # sns.lineplot(data=df[['sc_acc', 'tc_acc']], dashes=False, markers=True, ax=axes[0])
    axes[0].plot(df['ac_s'], label='$\mathcal{A}_{C}(S)$', marker='.')
    axes[0].plot(df['ac_t'], label='$\mathcal{A}_{C}(T)$', marker='x')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel(xlabel)
    axes[0].legend()

    # sns.lineplot(data=df[['sc_loss', 'tc_loss']], dashes=False, markers=True, ax=axes[1])
    axes[1].plot(df['lc_s'], label='$\mathcal{L}_{C}(S)$', marker='.')
    # axes[1].plot(df['lc_s2'], label='$\mathcal{L}_{C}(S2)$', marker='.')
    axes[1].plot(df['lc_t'], label='$\mathcal{L}_{C}(T)$', marker='x')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel(xlabel)
    axes[1].legend()
    
    f.suptitle('Classification')
    plt.show()

    
def wasserstein_plot(df, figsize=(14, 4), xlabel=''):
    """Plot Wassestein distance and gradient penalization."""
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    # gradient
    axes[0].plot(df['wd_grad'], marker='.')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('$\mathcal{L}_{grad}$')

    # wassestein loss
    axes[1].plot(-df['ewd_loss'], label='computed', marker='.')
    axes[1].plot(df['lwd'], label='estimated', marker='x')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('$\mathcal{L}_{wd}$')
    axes[1].legend()
    
    f.suptitle('Wasserstein')
    plt.show()

    
def wassestein_valid_plot(df, figsize=(7, 4), xlabel=''):
    f = plt.figure(figsize=figsize)
    # wassestein loss
    plt.plot(df['lwd'], marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('$\hat{\mathcal{L}}_{wd}$')
    plt.title('Wasserstein')
    plt.show()

    
def losses_plot(df, figsize=(7, 4), xlabel=''):
    """Plot total loss, l2 loss and source classification loss."""
    f, axes = plt.subplots(1, 1, figsize=figsize, sharex=True)
    """axes[0].plot(df['l2_loss'], marker='.')
    axes[0].set_ylabel('L2 norm')
    axes[0].set_xlabel(xlabel)"""

    axes.plot(df['loss'], label='$\mathcal{L}$', marker='.')
    axes.plot(df['lc_s1'], label='$\mathcal{L}_{C}(S)$', marker='x')
    # axes.plot(df['lwd'], label='$\hat{\mathcal{L}}_{wd}$', marker='x')
    axes.set_ylabel('Loss')
    axes.set_xlabel(xlabel)
    axes.legend()
    
    f.suptitle('Losses')
    plt.show()

    
def losses_valid_plot(df, figsize=(7, 4), xlabel=''):
    """Plot total loss, l2 loss and source classification loss."""
    f, axes = plt.subplots(1, 1, figsize=figsize, sharex=True)
    """axes[0].plot(df['l2_loss'], marker='.')
    axes[0].set_ylabel('L2 norm')
    axes[0].set_xlabel(xlabel)"""

    axes.plot(df['loss'], label='$\mathcal{L}$', marker='.')
    axes.plot(df['lc_s'], label='$\mathcal{L}_{C}(S)$', marker='x')
    # axes.plot(df['lwd'], label='$\hat{\mathcal{L}}_{wd}$', marker='x')
    axes.set_ylabel('Loss')
    axes.set_xlabel(xlabel)
    axes.legend()
    
    f.suptitle('Losses')
    plt.show()
