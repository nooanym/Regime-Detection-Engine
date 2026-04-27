"""yfinance-backed DataSource with 730d/1h guard and local parquet caching."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from rde.data.base import DataSource
from rde.data.cache import cache_path, is_cache_valid, read_cache, write_cache

logger = logging.getLogger(__name__)

_MAX_HOURLY_PERIOD_DAYS = 730


def _parse_period_days(period: str) -> int:
    """Parse a period string like '730d' into an integer number of days.

    Parameters
    ----------
    period : str
        Period string ending in 'd', e.g. "730d".

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If the format is not supported.

    """
    if period.endswith("d"):
        return int(period[:-1])
    raise ValueError(f"Unsupported period format: {period!r}. Expected '<N>d'.")


class YFinanceSource(DataSource):
    """DataSource backed by yfinance with local parquet caching.

    Parameters
    ----------
    cache_dir : Path | None
        Directory for parquet cache files. If None, caching is disabled.
    cache_max_age_seconds : int
        Maximum age (seconds) before a cached file is considered stale.

    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_max_age_seconds: int = 3600,
    ) -> None:
        """Initialise the source with optional cache directory and TTL."""
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache_max_age_seconds = cache_max_age_seconds

    def load(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Load OHLCV data for symbol. Serves from parquet cache when fresh.

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. "BTC-USD".
        period : str
            Lookback period, e.g. "730d". Must not exceed "730d" for interval="1h".
        interval : str
            Bar interval, e.g. "1h".

        Returns
        -------
        pd.DataFrame
            DataFrame with tz-aware DatetimeIndex and OHLCV columns.

        Raises
        ------
        ValueError
            If interval is "1h" and period exceeds the 730d retention cap.
        RuntimeError
            If yfinance returns empty data.

        """
        if interval == "1h":
            requested_days = _parse_period_days(period)
            if requested_days > _MAX_HOURLY_PERIOD_DAYS:
                raise ValueError(
                    f"Hourly data retention is capped at {_MAX_HOURLY_PERIOD_DAYS}d. "
                    f"Requested period={period!r} would silently return empty data from yfinance. "
                    "Use period='730d' or shorter."
                )

        if self._cache_dir is not None:
            path = cache_path(self._cache_dir, symbol, period, interval)
            if is_cache_valid(path, self._cache_max_age_seconds):
                logger.info("Cache hit: %s %s %s", symbol, period, interval)
                return read_cache(path)

        logger.info(
            "Downloading %s period=%s interval=%s from yfinance", symbol, period, interval
        )
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        # Flatten multi-level columns defensively (yfinance may return them for single symbols)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise RuntimeError(
                f"yfinance returned empty data for {symbol} "
                f"(period={period!r}, interval={interval!r}). "
                "Check the symbol and your internet connection."
            )

        df = df.dropna(subset=["Close"])

        if self._cache_dir is not None:
            path = cache_path(self._cache_dir, symbol, period, interval)
            write_cache(df, path)

        logger.info("Loaded %d rows for %s", len(df), symbol)
        return df
