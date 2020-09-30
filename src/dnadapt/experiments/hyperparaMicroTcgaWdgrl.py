import os
import time
import pandas as pd

from dnadapt.data.wdgrlLoader import load_data
from dnadapt.globals import datadir, logdir
from dnadapt.models.microTcgaModels import create_wdgrl_model, create_disc
from dnadapt.summary.writer import SummaryWriter
from dnadapt.training.wdgrl import train_model
from dnadapt.utils.data import random_split_data


def main():
    # data path
    src_path = os.path.join(datadir, 'microarray/train.npz')
    trg_path = os.path.join(datadir, 'tcga/train_eq.npz')

    src_train, src_valid = load_data(src_path, valid_size=12000)
    src_train, src_valid = random_split_data(src_valid, ratio=0.2)
    trg_train, trg_valid = load_data(trg_path, valid_size=0.2)

    # config values
    config = {
        'lambd': 1,
        'gamma': 10,
        'alpha1': 1e-3,
        'steps': 10,
        'alpha2': 1e-3,
        'epochs': 40,
        'bsize': 32,
        'patience': 7,
        'min_epoch': 10
    }

    src_size = 54675
    trg_size = 56602

    lambdas = [0, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
    gammas = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # experiment on lambda
    writer = SummaryWriter(os.path.join(logdir, f'micro_tcga_wdgrl_lambda_{date_time}'))
    run_hyper_param(config, src_size, trg_size, [src_train, trg_train], lambdas, 'lambd', writer,
                    valid_data=[src_valid, trg_valid])
    # experiment on gama
    config['lambd'] = 1
    writer.dir = os.path.join(logdir, f'micro_tcga_wdgrl_gama_{date_time}')
    run_hyper_param(config, src_size, trg_size, [src_train, trg_train], gammas, 'gama', writer,
                    valid_data=[src_valid, trg_valid])


def run_hyper_param(config, src_size, trg_size, train_data, params, param_name, writer, valid_data=None):
    meta_data = []
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
        watcher = train_model(model, train_data, valid_data=valid_data, disc=disc, **config)

        writer.write_watcher(watcher)
        del watcher
    df = pd.DataFrame(meta_data)
    df.to_csv(os.path.join(writer.dir, 'meta'))


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} (s)')
