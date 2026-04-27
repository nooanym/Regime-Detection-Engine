"""Abstract base class (FeatureTransformer) for feature transformations."""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureTransformer(ABC):
    """Abstract base for a single feature transformation.

    Each implementation receives a DataFrame, adds one or more columns,
    and returns the augmented DataFrame. Transformers must be pure — they
    must not mutate the input.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to df and return the augmented DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input price/feature DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional output columns; original columns preserved.

        """
        ...

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """Names of the columns this transformer adds."""
        ...
