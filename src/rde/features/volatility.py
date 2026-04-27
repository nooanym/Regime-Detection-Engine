"""Rolling-volatility feature transformer (std of log returns)."""

import pandas as pd

from rde.features.base import FeatureTransformer


class RollingVolatility(FeatureTransformer):
    """Rolling standard deviation of log returns as a volatility proxy.

    Parameters
    ----------
    window : int
        Rolling window size in bars. Default 24.

    Output column: ``volatility_w{window}``

    """

    def __init__(self, window: int = 24) -> None:
        """Store the rolling window size."""
        self._window = window

    @property
    def output_columns(self) -> list[str]:
        """Return ``['volatility_w{window}']``."""
        return [f"volatility_w{self._window}"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility_w{window} = log_return.rolling(window).std().

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'log_return' column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame plus rolling volatility column.

        """
        out = df.copy()
        col = f"volatility_w{self._window}"
        out[col] = df["log_return"].rolling(self._window).std()
        return out
