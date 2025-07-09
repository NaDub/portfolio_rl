import torch.nn as nn

from models.model_utils import fanin_init
from models.layers.layers import DenseLayer
from models.base import BaseModel
    
class Actor(BaseModel):
    def __init__(self, config, nb_action, nb_states):
        self.nb_action = nb_action
        self.nb_states = nb_states
        super().__init__(config)
        self.init_weights(self.config.init_w)

    def init_weights(self, init_w):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in self.head.modules():
                    module.weight.data.uniform_(-init_w, init_w)
                else:
                    module.weight.data = fanin_init(module.weight.data.size())
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def build_backbone(self):
        return nn.Sequential(
            DenseLayer(
                in_features=self.nb_states,
                out_features=self.config.hidden1,
                activation=nn.ReLU(),
                dropout=self.config.dropout
            ),
            DenseLayer(
                in_features=self.config.hidden1,
                out_features=self.config.hidden2,
                activation=nn.ReLU(),
                dropout=self.config.dropout
            )
        )

    def build_head(self):
        return DenseLayer(
                in_features=self.config.hidden2,
                out_features=self.nb_action,
                activation=nn.Tanh(),
                dropout=self.config.dropout
        )