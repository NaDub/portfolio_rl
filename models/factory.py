# models/factory.py

from models.mlp import MLPModel
from models.cnn import CNNModel
from models.rnn import RNNModel

def build_model(config):
    model_name = config.model.lower()

    if model_name == 'mlp':
        return MLPModel(config)
    elif model_name == 'cnn':
        return CNNModel(config)
    elif model_name == 'rnn':
        return RNNModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model}")
