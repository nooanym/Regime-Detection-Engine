"""Volume-based feature transformers."""

import numpy as np
import pandas as pd

from rde.features.base import FeatureTransformer


class LogVolume(FeatureTransformer):
    """Natural log of trading volume.

    Parameters
    ----------
    None

    Notes
    -----
    Applies np.log1p to avoid log(0) for any zero-volume bars.
    Output column: ``log_volume``.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log_volume column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``Volume`` column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional ``log_volume`` column.

        """
        out = df.copy()
        out["log_volume"] = np.log1p(df["Volume"].clip(lower=0))
        return out

    @property
    def output_columns(self) -> list[str]:
        """Names of columns produced by this transformer."""
        return ["log_volume"]


class RollingVolumeZScore(FeatureTransformer):
    """Rolling z-score of log-volume over a configurable window.

    Parameters
    ----------
    window : int
        Rolling window length in bars.

    Notes
    -----
    Z-score = (log_volume - rolling_mean) / rolling_std. The leading
    ``window - 1`` bars will be NaN and are dropped by FeaturePipeline.
    Output column: ``volume_zscore_w{window}``.
    """

    def __init__(self, window: int = 24) -> None:
        self.window = window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume_zscore_w{window} column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a ``Volume`` column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with an additional rolling volume z-score column.

        """
        out = df.copy()
        log_vol = np.log1p(df["Volume"].clip(lower=0))
        roll = log_vol.rolling(self.window)
        out[f"volume_zscore_w{self.window}"] = (log_vol - roll.mean()) / roll.std()
        return out

    @property
    def output_columns(self) -> list[str]:
        """Names of columns produced by this transformer."""
        return [f"volume_zscore_w{self.window}"]
