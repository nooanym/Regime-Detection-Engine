"""Log-return and smoothed-return feature transformers."""

import numpy as np
import pandas as pd

from rde.features.base import FeatureTransformer


class LogReturns(FeatureTransformer):
    """Compute log returns from Close prices.

    Output column: ``log_return``
    """

    @property
    def output_columns(self) -> list[str]:
        """Return ``['log_return']``."""
        return ["log_return"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log_return = log(Close / Close.shift(1)).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'Close' column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame plus 'log_return' column.

        """
        out = df.copy()
        out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return out


class SmoothedReturns(FeatureTransformer):
    """Rolling mean of log returns.

    Parameters
    ----------
    window : int
        Rolling window size in bars. Default 12.

    Output column: ``smoothed_return_w{window}``

    """

    def __init__(self, window: int = 12) -> None:
        """Store the rolling window size."""
        self._window = window

    @property
    def output_columns(self) -> list[str]:
        """Return ``['smoothed_return_w{window}']``."""
        return [f"smoothed_return_w{self._window}"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add smoothed_return_w{window} = log_return.rolling(window).mean().

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'log_return' column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame plus smoothed return column.

        """
        out = df.copy()
        col = f"smoothed_return_w{self._window}"
        out[col] = df["log_return"].rolling(self._window).mean()
        return out
