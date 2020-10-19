import os
import time
import pandas as pd

from dnadapt.data.wdgrlLoader import load_data
from dnadapt.globals import datadir, logdir
from dnadapt.models.microTcgaModels import create_wdgrl_model, create_disc
from dnadapt.summary.watcher import DataWatcher
from dnadapt.summary.writer import SummaryWriter
from dnadapt.training.fewShotWdgrl import train_model
from dnadapt.utils.data import random_split_data


def main():
    # data path
    src_path = os.path.join(datadir, 'microarray/train.npz')
    trg_path = os.path.join(datadir, 'tcga/train_eq.npz')

    src_train, src_valid = load_data(src_path, valid_size=0.2)
    trg_train, trg_valid = load_data(trg_path, valid_size=0.2)
    trg_train, trg_fs = random_split_data(trg_train, ratio=20)

    # config values
    config = {
        'lambd': 1,
        'gamma': 10,
        'alpha1': 1e-3,
        'alpha2': 1e-3,
        'eps': 1e-3,
        'epochs': 40,
        'steps': 10,
        'bsize': 32,
        'patience': 5,
        'min_epoch': 10,
    }

    src_size = 54675
    trg_size = 56602

    hyper_params = {
        'lambd': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 100.0],
        'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }

    run_exp([src_train, trg_train, trg_fs], config, hyper_params, src_size, trg_size, valid_data=[src_valid, trg_valid])


def run_exp(train_data, config, hyper_params, src_size, trg_size, valid_data=None):
    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # experiment on lambda
    key = 'lambd'
    writer = SummaryWriter(os.path.join(logdir, f'micro_tcga_fs_wdgrl_{key}_{date_time}'))
    run_hyper_param(config, src_size, trg_size, train_data, hyper_params[key], key, writer,
                    valid_data=valid_data)
    # experiment on gamma
    config[key] = 1.0
    key = 'gamma'
    writer.dir = os.path.join(logdir, f'micro_tcga_fs_wdgrl_{key}_{date_time}')
    run_hyper_param(config, src_size, trg_size, train_data, hyper_params[key], key, writer,
                    valid_data=valid_data)


def run_hyper_param(config, src_size, trg_size, train_data, params, param_name, writer, valid_data=None):
    meta_data = []
    min_watcher = DataWatcher(name="best_watcher")

    # train model
    for i, param in enumerate(params):
        # prepare data watching
        config[param_name] = param
        watcher_name = f'Watcher{i}'
        meta_data.append([param, watcher_name])

        # create model
        model = create_wdgrl_model(src_size, trg_size)
        # create discriminator
        disc = create_disc(500)

        # train model
        watcher, min_stats = train_model(model, train_data, valid_data=valid_data, disc=disc, **config)
        for key, val in min_stats.items():
            min_watcher.add_data(key, data=val)

        watcher.name = watcher_name
        writer.write_watcher(watcher)
        del [watcher, model, disc]
    writer.write_watcher(min_watcher)
    df = pd.DataFrame(meta_data)
    df.to_csv(os.path.join(writer.dir, 'meta'))


if __name__ == '__main__':
    main()
