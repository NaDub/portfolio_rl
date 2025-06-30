# data/transforms.py
import torch

class Normalize:
    def __init__(self, mean=None, std=None):
        """
        Normalizes input data using the provided or computed mean and standard deviation.

        Args:
            mean (float or None): Optional mean value. If None, computed from data.
            std (float or None): Optional standard deviation. If None, computed from data.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        Applies normalization to the input data.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized tensor.
        """
        if self.mean is None:
            self.mean = x.mean()
        if self.std is None:
            self.std = x.std()
        return (x - self.mean) / (self.std + 1e-8)

class SlidingWindow:
    def __init__(self, window_size, horizon):
        """
        Generates sliding windows from a time series.

        Args:
            window_size (int): Size of each sliding window.
            horizon (int): Forecasting horizon (number of steps ahead to predict).
        """
        self.window_size = window_size
        self.horizon = horizon

    def __call__(self, x):
        """
        Splits the input time series into overlapping windows.

        Args:
            x (Tensor): Input tensor of shape [T, D].

        Returns:
            Tensor: A stack of windows of shape [N_windows, window_size, D].
        """
        windows = []
        for i in range(len(x) - self.window_size - self.horizon + 1):
            x_window = x[i:i + self.window_size]
            windows.append(x_window)
        return torch.stack(windows)
    
class SlidingWindowSupervised:
    def __init__(self, window_size, horizon=1, multi_step=False):
        self.window_size = window_size
        self.horizon = horizon
        self.multi_step = multi_step

    def __call__(self, x, y):
        """
        Args:
            x (Tensor): [T, D_x]
            y (Tensor): [T, D_y]

        Returns:
            X: [N, window_size, D_x]
            Y: [N, D_y] or [N, horizon, D_y] depending on multi_step
        """
        X, Y = [], []
        T = len(x)
        for i in range(T - self.window_size - self.horizon + 1):
            x_window = x[i : i + self.window_size]

            if self.multi_step:
                y_target = y[i + self.window_size : i + self.window_size + self.horizon]
            else:
                y_target = y[i + self.window_size + self.horizon - 1]

            X.append(x_window)
            Y.append(y_target)

        return torch.stack(X), torch.stack(Y)
    
class Returns:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, x):
        """
        Computes simple returns: (x[t] - x[t-1]) / x[t-1]

        Args:
            x (Tensor or ndarray): Input of shape [T, D]

        Returns:
            Tensor: Returns of shape [T-1, D]
        """
        returns = (x[1:] - x[:-1]) / (x[:-1] + self.eps)
        return returns
    
class LogReturns:
    def __call__(self, x):
        """
        Computes log returns: log(x[t]) - log(x[t-1])

        Args:
            x (Tensor): Input of shape [T, D]

        Returns:
            Tensor: Log returns of shape [T-1, D]
        """
        return torch.log(x[1:] + 1e-8) - torch.log(x[:-1] + 1e-8)

class Compose:
    def __init__(self, transforms):
        """
        Compose several transforms together.

        Args:
            transforms (list of callables): List of transformations to apply sequentially.
        """
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
       
def get_transform(transform, config):
    if transform=='sliding_window':
        return SlidingWindow(config.window_size, config.horizon)
    elif transform=='log_returns':
        return LogReturns()
    elif transform=='returns':
        return Returns()
    elif transform=='normalize':
        return Normalize()
    else: 
        raise ValueError(f'Unknown transform: {transform}')
    
def build_compose(config):
    transforms = [get_transform(transform, config) for transform in config.transforms]
    return Compose(transforms)
