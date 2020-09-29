# code from
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import copy
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: None (model is not saved)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        # save the best model
        self.train_acc = 0
        self.valid_acc = 0
        self.best_param = None

    def __call__(self, model, val_loss, train_acc, valid_acc):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.train_acc = train_acc
            self.valid_acc = valid_acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def stop(self):
        return self.counter >= self.patience

    def save_checkpoint(self, val_loss, model):
        """Try to save model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')

        if self.path is not None:
            # save model if path given
            self.best_param = copy.deepcopy(model.state_dict())
            print('Saving model ...')
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
