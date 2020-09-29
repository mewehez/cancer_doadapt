import os
import pandas as pd


class SummaryWriter(object):
    def __init__(self, dir_path):
        self.dir = dir_path

    def write_watcher(self, watcher):
        watcher.save(self.dir)

    def write_watchers(self, watchers):
        meta = [[key, watcher.name] for key, watcher in watchers.items()]
        for param, watcher in watchers.items():
            watcher.save(self.dir)

        df = pd.DataFrame(meta)
        df.to_csv(os.path.join(self.dir, 'meta'))
