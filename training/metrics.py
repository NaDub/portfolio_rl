import torch

class MAE:
    def __call__(self, pred, true):
        return torch.mean(torch.abs(pred.detach() - true.detach())).item()

class MSE:
    def __call__(self, pred, true):
        return torch.mean((pred.detach() - true.detach()) ** 2).item()

class RMSE:
    def __call__(self, pred, true):
        mse = torch.mean((pred.detach() - true.detach()) ** 2)
        return torch.sqrt(mse).item()

class MAPE:
    def __call__(self, pred, true):
        return torch.mean(torch.abs((pred.detach() - true.detach()) / (true.detach() + 1e-8))).item()

class MSPE:
    def __call__(self, pred, true):
        return torch.mean(((pred.detach() - true.detach()) / (true.detach() + 1e-8)) ** 2).item()

class RSE:
    def __call__(self, pred, true):
        pred, true = pred.detach(), true.detach()
        return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))

class CORR:
    def __call__(self, pred, true):
        pred, true = pred.detach(), true.detach()
        u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
        d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
        return (u / d).mean().item()

def get_metrics(name):
    name = name.lower()
    if name == 'mae':
        return MAE()
    elif name == 'mse':
        return MSE()
    elif name == 'rmse':
        return RMSE()
    elif name == 'mape':
        return MAPE()
    elif name == 'mspe':
        return MSPE()
    elif name == 'rse':
        return RSE()
    elif name == 'corr':
        return CORR()
    else:
        raise ValueError(f"Unknown metric: {name}")
