import numpy as np


def _combine_dict(dict1, dict2):
    for key, val in dict2.items():
        if key not in dict1.keys():
            dict1[key] = [val]
        else:
            dict1[key].append(val)


def combine_dict(in_dict, _dict):
    out_dict = dict(in_dict)
    _combine_dict(out_dict, _dict)
    return out_dict


class StatsData(object):
    def __init__(self):
        self.data_run = {}
        self.data = {}

    def update(self, data):
        _combine_dict(self.data_run, data)

    def snap(self):
        _combine_dict(self.data, {key: np.mean(val) for (key, val) in self.data_run.items()})
        self.data_run = {}

    def reset(self):
        self.data_run = {}
        self.data = {}
