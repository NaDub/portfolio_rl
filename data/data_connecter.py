import torch
from torch.utils.data import DataLoader

from data.data_provider.binance import BinanceClient
from data.transforms import Returns, SlidingWindowSupervised
from data.dataset import TimeSeriesDataset

def get_binance(config):
    df = BinanceClient().get_ohlc(limit=1000, symbol="BTCUSDT", interval="1h")
    x = torch.tensor(df[['open', 'high', 'low']].values, dtype=torch.float32)
    y = torch.tensor(df['close'].values, dtype=torch.float32)

    x, y = Returns()(x), Returns()(y)
    x, y = SlidingWindowSupervised(window_size=50, horizon=5, multi_step=True)(x=x, y=y)

    dataset = TimeSeriesDataset(x, y, transform=None)
    train_set, val_set, test_set = dataset.temporal_split()

    # print("dataset shape of x:", dataset[:][0].shape)
    # print("data:", train_set[0])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # for batch_X, batch_y in train_loader:
    #     print("Input shape:", batch_X.shape)
    #     print("Target shape:", batch_y.shape)

    return train_loader, val_loader, test_loader