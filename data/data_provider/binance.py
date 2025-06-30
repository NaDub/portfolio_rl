import requests
import pandas as pd
from datetime import datetime

class BinanceClient:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    def get_ohlc(self, limit=100, symbol="BTCUSDT", interval="1h"):
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        columns = [
            'time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(data, columns=columns)
        df[columns] = df[columns].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df = df[['time', 'open', 'high', 'low', 'close']]
        return df
