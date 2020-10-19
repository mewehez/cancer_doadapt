from torch import nn

from dnadapt.globals import device
from dnadapt.models.discriminator import Discriminator
from dnadapt.models.wdgrlModels import MTWdgrlNet
from dnadapt.utils.functions import init_weights


def create_disc(n_input):
    disc_model = nn.Sequential(
        nn.Linear(n_input, 100),
        nn.ReLU(),
        nn.Linear(100, 2)
    )
    init_weights(disc_model)  # Init layer weights
    disc_model.to(device)
    disc_loss = nn.CrossEntropyLoss()  # classifier loss function
    discriminator = Discriminator(disc_model, disc_loss)
    return discriminator


def create_wdgrl_model(src_size, trg_size):
    gen_s = nn.Sequential(
        nn.Linear(src_size, 2000),
        nn.BatchNorm1d(2000, momentum=0.3),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2000, 500),
        nn.ReLU()
    )
    gen_t = nn.Sequential(
        nn.Linear(trg_size, 2000),
        nn.BatchNorm1d(2000, momentum=0.3),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2000, 500),
        nn.ReLU()
    )
    # gen = nn.Sequential(
    #     nn.Linear(2000, 500),
    #     nn.ReLU()
    # )
    critic = nn.Sequential(
        nn.Linear(500, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    disc = nn.Sequential(
        nn.Linear(500, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    model = MTWdgrlNet(gen_s, gen_t, disc, critic)
    init_weights(model)
    return model.to(device)
