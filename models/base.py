# models/base.py
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone()
        self.head = self.build_head()

    @abstractmethod
    def build_backbone(self):
        """Define extractor caracteristics"""
        pass

    @abstractmethod
    def build_head(self):
        """Define head (output layer)"""
        pass

    def forward(self, *x):
        features = self.backbone(*x)
        output = self.head(features)
        return output