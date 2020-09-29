import os
from dnadapt.experiments import toyWdgrl, toyFewShotWdgrl, \
    microTcgaWdgrl, microTcgaFewShotWdgrl
from dnadapt.data.toy import create_data_file
from dnadapt.globals import datadir

"""
https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
"""


if __name__ == '__main__':
    microTcgaWdgrl.main()
    
    # data_config = {
    #     'src_size': 5000,
    #     'trg_size': 1000,
    #     'src_pcts': [0.37, 0.63],
    #     'trg_pcts': [0.37, 0.63],
    #     'src_centers': [[0, 0], [0, 10]],
    #     'trg_centers': [[50, -20], [50, -10]]
    # }
    # 
    # create_data_file(os.path.join(datadir, 'toy'), **data_config)
