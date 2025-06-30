import argparse
import toml
from types import SimpleNamespace

def load_config(*paths):
    config = {}
    for path in paths:
        cfg = toml.load(path)
        config.update(cfg)
    return SimpleNamespace(**config)

def get_parser():
    parser = argparse.ArgumentParser(description="Training time series model")
    parser.add_argument('--type', type=str, default='DL', help='RL or DL')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config_name', type=str, nargs='+', default=['base.toml'])

    return parser.parse_args()

def select_toml(args):
    if args.type == 'DL':
        base = f'configs/forecast'
    if args.type == 'RL':
        base = f'configs/rl'
    return [f"{base}/{name}" for name in args.config_name]