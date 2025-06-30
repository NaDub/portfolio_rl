# training/losses.py
import torch.nn as nn

def get_loss(name):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss: {name}")
