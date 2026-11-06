"""Download and cache daily prices, then turn them into log returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fetch_prices(tickers: list[str], start: str, end: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """Adjusted close prices (dates x tickers). Cached to CSV so reruns are offline and reproducible."""
    cache = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    import yfinance as yf  # imported lazily so tests don't need network access

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().iloc[1:]
