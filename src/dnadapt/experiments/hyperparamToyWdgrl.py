import os
import time
import pandas as pd

from dnadapt.data.wdgrlLoader import load_data
from dnadapt.globals import datadir, logdir
from dnadapt.models.toyModels import create_wdgrl_model, create_disc
from dnadapt.summary.writer import SummaryWriter
from dnadapt.training.wdgrl import train_model


def main():
    src_path = os.path.join(datadir, 'toy', 'src', 'train.npz')
    trg_path = os.path.join(datadir, 'toy', 'trg', 'train.npz')

    src_train, src_valid = load_data(src_path, valid_size=0.2)
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
        'patience': 1,
        'min_epoch': 5
    }

    n_input = src_train[0].shape[1]
    n_class = 2
    n_hidden = [2]

    lambdas = [0, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
    gammas = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # experiment on lambda
    writer = SummaryWriter(os.path.join(logdir, f'toy_wdgrl_lambda_{date_time}'))
    run_hyper_param(config, n_class, n_hidden, n_input, [src_train, trg_train], lambdas, 'lambd', writer,
                    valid_data=[src_valid, trg_valid])
    # experiment on gama
    config['lambd'] = 1
    writer.dir = os.path.join(logdir, f'toy_wdgrl_gama_{date_time}')
    run_hyper_param(config, n_class, n_hidden, n_input, [src_train, trg_train], gammas, 'gama', writer,
                    valid_data=[src_valid, trg_valid])


def run_hyper_param(config, n_class, n_hidden, n_input, train_data, params, param_name, writer, valid_data=None):
    meta_data = []
    # train model
    for i, param in enumerate(params):
        # prepare data watching
        config[param_name] = param
        watcher_name = f'Watcher{i}'
        meta_data.append([param, watcher_name])

        # create WDGRL model
        model = create_wdgrl_model(n_class, n_hidden, n_input)
        # create discriminator
        disc = create_disc(n_hidden)

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
