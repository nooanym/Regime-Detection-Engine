"""FeaturePipeline: compose transformers and produce a clean feature matrix."""

import logging

import pandas as pd

from rde.features.base import FeatureTransformer

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Compose multiple FeatureTransformers and drop NaN rows at the end.

    The pipeline is the single authoritative place where features are produced.
    Models, inference, and evaluation modules must never compute features themselves.

    Parameters
    ----------
    transformers : list[FeatureTransformer]
        Ordered list of transformers to apply sequentially.

    """

    def __init__(self, transformers: list[FeatureTransformer]) -> None:
        """Store the ordered list of transformers."""
        self._transformers = transformers

    @property
    def output_columns(self) -> list[str]:
        """All feature column names produced by this pipeline."""
        cols: list[str] = []
        for t in self._transformers:
            cols.extend(t.output_columns)
        return cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformers in order, then drop NaN rows.

        Parameters
        ----------
        df : pd.DataFrame
            Raw price DataFrame.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame with no NaN rows.

        Raises
        ------
        ValueError
            If any output feature column is entirely NaN after transformation.

        """
        result = df.copy()
        for transformer in self._transformers:
            result = transformer.transform(result)

        result = result.dropna()

        for col in self.output_columns:
            if col in result.columns and result[col].isna().all():
                raise ValueError(
                    f"Feature column '{col}' is entirely NaN after pipeline. "
                    "Check transformer configuration and input data."
                )

        logger.debug(
            "FeaturePipeline produced %d rows with features: %s",
            len(result),
            self.output_columns,
        )
        return result
