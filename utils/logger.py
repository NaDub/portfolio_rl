import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir="logs", run_name=None):
        if run_name is None:
            run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_path = os.path.join(log_dir, run_name)
        os.makedirs(self.full_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.full_path)

    def log_scalar(self, name, value, step, prefix=None):
        """Log a scalar value (e.g. loss, accuracy)."""
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_scalar(tag, value, step)

    def log_figure(self, name, figure, step, prefix=None):
        """Log a matplotlib figure."""
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_figure(tag, figure, step)

    def log_image(self, name, image_tensor, step, prefix=None):
        """Log a [C, H, W] tensor as image (e.g. output from a model)."""
        tag = f"{prefix}/{name}" if prefix else name
        self.writer.add_image(tag, image_tensor, step)

    def close(self):
        self.writer.close()