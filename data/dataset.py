# data/dataset.py
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Custom PyTorch Dataset for time series data.

        Args:
            data (Tensor or ndarray): Input data of shape [N, T, D],
                where N = number of samples, T = time steps, D = features.
            targets (Tensor or ndarray): Target values of shape [N, ...].
            transform (callable, optional): Optional transform to apply on each input sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample and its corresponding target.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input sample, target)
        """
        x = self.data[idx]
        #print(f"[DEBUG] x: x.shape = {x.shape}")
        y = self.targets[idx]
        #print(f"[DEBUG] Before transform: x.shape = {x.shape}")
        if self.transform:
            x = self.transform(x)
            #print(f"[DEBUG] After transform:  x.shape = {x.shape}")
        return x, y
    
    def temporal_split(self, train_ratio=0.7, val_ratio=0.15):
        total = len(self.data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_set = Subset(self, list(range(0, train_end)))
        val_set   = Subset(self, list(range(train_end, val_end)))
        test_set  = Subset(self, list(range(val_end, total)))

        return train_set, val_set, test_set
