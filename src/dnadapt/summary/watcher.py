import os
import numpy as np

from dnadapt.utils.utils import folder_if_not_exist


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


class DataWatcher(object):
    _counter = 0

    def __init__(self, name=None):
        if name is None or len(name) == 0:
            name = f'Watcher{DataWatcher._counter}'
            DataWatcher._counter += 1
        self.name = name
        self.data = {}

    def add_data(self, name, data=None):
        if name not in self.data.keys():
            self.data[name] = StatsData()

        if data is not None:
            _combine_dict(self.data[name].data, data)

    def reset(self):
        self.data = {}

    def save(self, path):
        fold_path = folder_if_not_exist(os.path.join(path, self.name))

        for name, stats_data in self.data.items():
            np.savez_compressed(os.path.join(fold_path, name), **stats_data.data)  # scatter the dict
