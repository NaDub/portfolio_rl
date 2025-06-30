# models/mlp.py
from models.base import BaseModel
from models.layers.layers import DenseLayer
import torch.nn as nn

class MLPModel(BaseModel):
    def build_backbone(self):
        return nn.Sequential(
            DenseLayer(
                in_features=self.config.input_dim,
                out_features=self.config.hidden_dim,
                activation=nn.ReLU(),
                dropout=self.config.dropout
            ),
            DenseLayer(
                in_features=self.config.hidden_dim,
                out_features=self.config.hidden_dim,
                activation=nn.ReLU(),
                dropout=self.config.dropout
            )
        )

    def build_head(self):
        return nn.Linear(self.config.hidden_dim, self.config.output_dim)

    
# class MLPModelOld(BaseModel):
#     def build_backbone(self):
#         return nn.Sequential(
#             nn.Linear(self.config.input_dim, self.config.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
#             nn.ReLU()
#         )

#     def build_head(self):
#         return nn.Linear(self.config.hidden_dim, self.config.output_dim)