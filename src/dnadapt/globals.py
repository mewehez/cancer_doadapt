import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# folder paths
_filepath = os.path.dirname(os.path.abspath(__file__))

logdir = os.path.join(_filepath, '../../log/')
modeldir = os.path.join(_filepath, '../../models/')
datadir = os.path.join(_filepath, '../../../data/')
