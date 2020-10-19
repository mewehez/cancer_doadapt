import os
import time

from dnadapt.data.wdgrlLoader import load_data
from dnadapt.globals import datadir, logdir
from dnadapt.models.toyModels import create_wdgrl_model, create_disc
from dnadapt.training.fewShotWdgrl import train_model
from dnadapt.utils.data import random_split_data


def main():
    src_path = os.path.join(datadir, 'toy', 'src', 'train.npz')
    trg_path = os.path.join(datadir, 'toy', 'trg', 'train.npz')

    src_train, src_valid = load_data(src_path, valid_size=0.2)
    trg_train, trg_valid = load_data(trg_path, valid_size=0.2)
    trg_train, trg_fs = random_split_data(trg_train, ratio=20)

    # config values
    config = {
        'lambd': 1,
        'gamma': 10,
        'alpha1': 1e-3,
        'steps': 10,
        'alpha2': 1e-3,
        'epochs': 40,
        'bsize': 32,
        'patience': 5,
        'min_epoch': 10,
        'eps': 1e-3
    }

    n_input = src_train[0].shape[1]
    n_class = 2
    n_hidden = [2]

    # create WDGRL model
    model = create_wdgrl_model(n_class, n_hidden, n_input)
    # create discriminator
    disc = create_disc(n_hidden)

    watcher, _ = train_model(model, [src_train, trg_train, trg_fs], valid_data=[src_valid, trg_valid],
                             disc=disc, **config)
    # save stats
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    watcher.name = f'toy_fs_wdgrl_{date_time}'
    watcher.save(logdir)


if __name__ == '__main__':
    main()
