"""Abstract base class (DataSource) for market data loaders."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract base class for market data sources."""

    @abstractmethod
    def load(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Load OHLCV data for a given symbol, period, and interval.

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. "BTC-USD".
        period : str
            Lookback period, e.g. "730d".
        interval : str
            Bar interval, e.g. "1h".

        Returns
        -------
        pd.DataFrame
            DataFrame with tz-aware DatetimeIndex and columns
            {Open, High, Low, Close, Volume}.

        """
        ...
