from torch import nn, optim

from dnadapt.globals import device
from dnadapt.models.discriminator import Discriminator
from dnadapt.models.wdgrlModels import WDGRLNet
from dnadapt.utils.functions import init_linear_weight


def create_disc(n_hidden):
    disc_model = nn.Linear(n_hidden[-1], 2)
    init_linear_weight(disc_model)  # Init layer weights
    disc_model.to(device)
    disc_loss = nn.CrossEntropyLoss()  # classifier loss function
    discriminator = Discriminator(disc_model, disc_loss)
    return discriminator


def get_ftr_extractor(input_size, n_hidden, act=nn.ReLU(), cte=0.1):
    layers = []
    for size in n_hidden:
        layer = nn.Linear(input_size, size)
        # Init layer weights
        init_linear_weight(layer, cte=cte)

        # add the layer and activation
        layers.append(layer)
        layers.append(act)
        input_size = size

    # return sequential model
    return nn.Sequential(*layers)


def get_class_clf(input_size, outsize, cte=0.1):
    layer = nn.Linear(input_size, outsize)
    # Init layer weights
    init_linear_weight(layer, cte=cte)
    return layer


def get_critic(input_size, crit_hidden, cte=0.1):
    layers = []
    # input layer
    layer = nn.Linear(input_size, crit_hidden)
    init_linear_weight(layer, cte=cte)
    layers.append(layer)
    # output layer
    layer = nn.Linear(crit_hidden, 1)
    init_linear_weight(layer, cte=cte)
    layers.append(layer)
    # return sequential model
    return nn.Sequential(*layers)


def create_wdgrl_model(n_class, n_hidden, n_input):
    ftr_extractor = get_ftr_extractor(n_input, n_hidden)
    dom_critic = get_critic(n_hidden[-1], 10)
    class_clf = get_class_clf(n_hidden[-1], n_class)
    model = WDGRLNet(ftr_extractor, class_clf, dom_critic).to(device)
    return model
