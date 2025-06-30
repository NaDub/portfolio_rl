import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """Fully connected layer avec activation et dropout optionnel."""
    def __init__(self, in_features, out_features, activation=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class Conv1DLayer(nn.Module):
    """1D convolution pour time-series"""
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class LSTMLayer(nn.Module):
    """LSTM wrapper"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out
    
class SimpleRNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return out

class ConcatDenseBlock(nn.Module):
    """
    Backbone that takes a state vector x and an action vector a,
    encodes x, concatenates it with a, and processes the result through
    another dense layer. Useful for critics in actor-critic architectures.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.encoder = DenseLayer(state_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)
        self.concat_layer = DenseLayer(hidden_dim + action_dim, hidden_dim, activation=nn.ReLU(), dropout=dropout)

    def forward(self, x, a):
        x = self.encoder(x)
        x = torch.cat([x, a], dim=1)
        x = self.concat_layer(x)
        return x
    
class Flatten(nn.Module):
    """Flatten utilitaire"""
    def forward(self, x):
        return torch.flatten(x, start_dim=1)
