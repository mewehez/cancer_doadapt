import copy
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=1e-3):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 1e-3
        """
        self.patience = patience
        self.delta = delta
        self._counter = 0
        self.min_vals = None
        self._early_stop = False
        self.min_score = np.Inf

    def __call__(self, score, vals=None):
        if score < self.min_score + self.delta:
            self._counter = 0
            self._early_stop = False
            self.min_score = score
            self.min_vals = vals
        else:
            self._counter += 1
            print(f'EarlyStopping counter: {self._counter} out of {self.patience}')
            if self._counter >= self.patience:
                self._early_stop = True
    
    def stop(self):
        return self._early_stop
