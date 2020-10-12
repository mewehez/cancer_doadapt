import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
sns.set(style="whitegrid", palette="muted", color_codes=True)

# https://matplotlib.org/tutorials/intermediate/legend_guide.html


def join_data(path, file, meta, index_col=0):
    dt = {}
    index = []
    for (i, conf) in enumerate(meta.values):
        arr = np.load(os.path.join(path, conf[1], file+'.npz'), allow_pickle=True)
        # arr = arr['arr_0'].item()
        for key, val in arr.items():
            if key not in dt.keys():
                dt[key] = []
            dt[key].append(val[-1])
        index.append(np.log10(conf[0] + 1e-20))  # add small value to avoid division by zero error
    d = pd.DataFrame(dt, index=index)
    return d.iloc[1:]


def disc_plot(df, figsize=(14, 5), xlabel=''):
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


def src_trg_plot(df_train, df_valid, figsize=(14, 5), xlabel=''):
    """Plot source and target losses and accuracies."""
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    
    # prepare legend
    src = mlines.Line2D([], [], color='blue',  marker='.', label="$\mathcal{A}_{C}(S)$")
    trg = mlines.Line2D([], [], color='orange',  marker='x', label="$\mathcal{A}_{C}(T)$")
    trg_f = mlines.Line2D([], [], color='green',  marker='+', label="$\mathcal{A}_{C}(F)$")
    # p_lines = mlines.Line2D([], [], color='k', label="Training")
    # d_lines = mlines.Line2D([], [], linestyle='--', color='k', label="Validation")
    # first_legend = axes[0].legend(handles=[p_lines, d_lines], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0].plot(df_train['ac_s'], color='blue', marker='.')
    axes[0].plot(df_train['ac_t'], color='orange', marker='x')
    axes[0].plot(df_train['ac_f'], color='green', marker='+')
    
    axes[0].plot(df_valid['ac_s'], color='blue', linestyle= '--', marker='.')
    axes[0].plot(df_valid['ac_t'], color='orange', linestyle= '--', marker='x')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel(xlabel)
    axes[0].legend(handles=[src, trg, trg_f])
    
    # prepare legend
    src = mlines.Line2D([], [], color='blue',  marker='.', label="$\mathcal{L}_{C}(S)$")
    trg = mlines.Line2D([], [], color='orange',  marker='x', label="$\mathcal{L}_{C}(T)$")
    trg_f = mlines.Line2D([], [], color='green',  marker='+', label="$\mathcal{L}_{C}(F)$")

    axes[1].plot(df_train['lc_s'], color='blue', marker='.')
    axes[1].plot(df_train['lc_t'], color='orange', marker='x')
    axes[1].plot(df_train['lc_f'], color='green', marker='+')
    
    axes[1].plot(df_valid['lc_s'], color='blue', linestyle= '--', marker='.')
    axes[1].plot(df_valid['lc_t'], color='orange', linestyle='--', marker='x')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel(xlabel)
    axes[1].legend(handles=[src, trg, trg_f])
    
    f.suptitle('Classification')
    plt.show()

    
def wasserstein_plot(df_train, df_valid, figsize=(14, 5), xlabel=''):
    """Plot Wassestein distance and gradient penalization."""
    f, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    # gradient
    axes[0].plot(df_train['lgrad'], marker='.')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('$\mathcal{L}_{grad}$')

    # wassestein loss
    # prepare legend
    lwd_s = mlines.Line2D([], [], color='blue',  marker='.', label="$\mathcal{L}_{wd}^{*}$")
    lwd_e = mlines.Line2D([], [], color='orange',  marker='x', label="$\hat{\mathcal{L}}_{wd}$")
    
    lwd_ = -df_train['lwd_'] + (10**df_train.index) * df_train['lgrad']
    axes[1].plot(lwd_, label='$\mathcal{L}_{wd}^{*}$', color='blue', marker='.')
    axes[1].plot(df_train['lwd'], color='orange', marker='x')
    axes[1].plot(df_valid['lwd'], color='orange', linestyle='--', marker='x')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('$\mathcal{L}_{wd}$')
    axes[1].legend(handles=[lwd_s, lwd_e])
    
    f.suptitle('Wasserstein')
    plt.show()

    
def losses_plot(df_train, df_valid, figsize=(7, 5), xlabel=''):
    """Plot total loss, l2 loss and source classification loss."""
    f, axes = plt.subplots(1, 1, figsize=figsize, sharex=True)
    
    # prepare legend
    total = mlines.Line2D([], [], color='blue',  marker='.', label="$\mathcal{L}$")
    src = mlines.Line2D([], [], color='orange',  marker='x', label="$\mathcal{L}_{C}(S)$")
    trg_f = mlines.Line2D([], [], color='green',  marker='+', label="$\mathcal{L}_{C}(F)$")

    axes.plot(df_train['loss'], label='$\mathcal{L}$', color='blue', marker='.')
    axes.plot(df_train['lc_s'], label='$\mathcal{L}_{C}(S)$', color='orange', marker='x')
    axes.plot(df_train['lc_f'], label='$\mathcal{L}_{C}(F)$', color='green', marker='+')

    axes.plot(df_valid['loss'], color='blue', linestyle='--', marker='.')
    axes.plot(df_valid['lc_s'], color='orange', linestyle='--', marker='x')
    axes.set_ylim((-0.1, 1.85))
    axes.set_ylabel('Loss')
    axes.set_xlabel(xlabel)
    axes.legend(handles=[total, src, trg_f])
    
    f.suptitle('Losses')
    plt.show()
