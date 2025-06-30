# training/optimizer.py
import torch.optim as optim

def get_optimizer(model, config):
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
