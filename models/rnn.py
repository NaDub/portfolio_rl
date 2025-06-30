import torch.nn as nn
from models.base import BaseModel
from models.layers.layers import SimpleRNNBlock

class RNNModel(BaseModel):
    def build_backbone(self):
        return SimpleRNNBlock(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional
        )

    def build_head(self):
        factor = 2 if self.config.bidirectional else 1
        return nn.Linear(self.config.hidden_dim * factor, self.config.output_dim)

    def forward(self, x):
        features = self.backbone(x)
        last_output = features[:, -1, :]
        return self.head(last_output)

    # def forward(self, x):
    # features = self.backbone(x)        # [B, T, H]
    # last_output = features[:, -1, :]   # [B, H]
    # output = self.head(last_output)    # [B, output_dim]
    # return output
