import torch
from torch.utils.data import Dataset

from dnadapt.globals import device


class CustomDataset(Dataset):

    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.as_tensor(self.X[idx]).float().to(device)
        y = torch.as_tensor(self.Y[idx]).long().to(device)

        return [x, y]

    def tensors(self):
        return torch.as_tensor(self.X).float().to(device), \
               torch.as_tensor(self.Y).long().to(device)
