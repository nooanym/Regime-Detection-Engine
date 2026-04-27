"""Parquet cache helpers: path generation, staleness check, read, and write."""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cache_path(cache_dir: Path, symbol: str, period: str, interval: str) -> Path:
    """Return the parquet path for the given (symbol, period, interval) triple.

    Parameters
    ----------
    cache_dir : Path
        Root directory for cached files.
    symbol : str
        Ticker symbol.
    period : str
        Lookback period string, e.g. "730d".
    interval : str
        Bar interval string, e.g. "1h".

    Returns
    -------
    Path
        Full path to the parquet file.

    """
    safe = symbol.replace("/", "-")
    return cache_dir / f"{safe}_{period}_{interval}.parquet"


def is_cache_valid(path: Path, max_age_seconds: int = 3600) -> bool:
    """Return True if path exists and its mtime is within max_age_seconds of now.

    Parameters
    ----------
    path : Path
        File path to check.
    max_age_seconds : int
        Maximum allowed age in seconds.

    Returns
    -------
    bool

    """
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - mtime
    return age < timedelta(seconds=max_age_seconds)


def read_cache(path: Path) -> pd.DataFrame:
    """Read a cached parquet file and validate its index type.

    Parameters
    ----------
    path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the cached file does not have a DatetimeIndex.

    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Cached file {path} does not have a DatetimeIndex")
    logger.debug("Cache read: %d rows from %s", len(df), path)
    return df


def write_cache(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    path : Path
        Destination parquet path.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.debug("Cache write: %d rows to %s", len(df), path)
