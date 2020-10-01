import os

from dnadapt.data.wdgrlLoader import load_data, create_test_data_loader
from dnadapt.globals import datadir
from dnadapt.models.microTcgaModels import create_wdgrl_model, create_disc
from dnadapt.training.wdgrl import train_model, make_model_tester
from dnadapt.utils.data import random_split_data


def main():
    # data path
    src_path = os.path.join(datadir, 'microarray/train.npz')
    trg_path = os.path.join(datadir, 'tcga/train_eq.npz')

    src_train, src_valid = load_data(src_path, valid_size=0.2)
    # src_train, src_valid = random_split_data(src_valid, ratio=0.2)
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
        'patience': 3,
        'min_epoch': 5
    }
    
    src_size = 54675
    trg_size = 56602
    # create model
    model = create_wdgrl_model(src_size, trg_size)
    # create discriminator
    disc = create_disc(500)

    # train model
    train_model(model, [src_train, trg_train], valid_data=[src_valid, trg_valid], disc=disc, **config)

    """Test model"""
    src_test_path = os.path.join(datadir, 'microarray/test.npz')
    trg_test_path = os.path.join(datadir, 'tcga/test.npz')

    src_test_loader, trg_test_loader = create_test_data_loader(src_test_path, trg_test_path, 32)

    # test model
    model_tester = make_model_tester(model)
    src_test_acc = model_tester(src_test_loader)
    trg_test_acc = model_tester(trg_test_loader, src=False)
    print(f'source accuracy: {src_test_acc}')
    print(f'target accuracy: {trg_test_acc}')


if __name__ == '__main__':
    main()
