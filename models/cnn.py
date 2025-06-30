from models.base import BaseModel
from models.layers.layers import Conv1DLayer, Flatten
import torch.nn as nn

class CNNModel(BaseModel):
    """ Input format: [N, C, L] (Batch, Channels, Lenght)"""
    def build_backbone(self):
        return nn.Sequential(
            Conv1DLayer(
                in_channels=self.config.input_channels,
                out_channels=self.config.hidden_channels,
                kernel_size=self.config.kernel_size,
                activation=nn.ReLU(),
                dropout=self.config.get("dropout", 0.0)
            ),
            Conv1DLayer(
                in_channels=self.config.hidden_channels,
                out_channels=self.config.hidden_channels,
                kernel_size=self.config.kernel_size,
                activation=nn.ReLU(),
                dropout=self.config.get("dropout", 0.0)
            ),
            Flatten()
        )

    def build_head(self):
        return nn.Linear(self.config.flat_features, self.config.output_dim)
