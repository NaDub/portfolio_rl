import torch.nn as nn

from models.model_utils import fanin_init
from models.base import BaseModel
from models.layers.layers import ConcatDenseBlock
    
class Critic(BaseModel):
    """
    Modular critic model for actor-critic algorithms like DDPG or TD3.
    Takes both state and action as input, processes state first, then
    concatenates with action before final prediction.
    """
    def __init__(self, config, nb_action, nb_states):
        self.nb_action = nb_action
        self.nb_states = nb_states
        super().__init__(config)
        self.init_weights(self.config.init_w)

    def init_weights(self, init_w):
        """Custom weight initialization for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in self.head.modules():
                    module.weight.data.uniform_(-init_w, init_w)
                else:
                    module.weight.data = fanin_init(module.weight.data.size())
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def build_backbone(self):
        return ConcatDenseBlock(
            state_dim=self.nb_states,
            action_dim=self.nb_action,
            hidden_dim=self.config.hidden2,
            dropout=self.config.dropout
        )

    def build_head(self):
        return nn.Linear(self.config.hidden2, 1)
    
