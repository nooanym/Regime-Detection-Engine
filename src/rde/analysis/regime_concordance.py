"""Regime concordance analysis across multiple assets (Phase 34).

Quantifies how synchronised regime sentiment is across a universe of assets,
extending cross_asset.py with rolling, pairwise, and global concordance metrics.

Public API
----------
ConcordanceConfig
PairwiseConcordance
ConcordanceResult
compute_concordance
concordance_heatmap_data
rolling_concordance_series
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConcordanceConfig:
    """Configuration for regime concordance computation.

    Attributes
    ----------
    min_common_bars : int
        Minimum number of common bars required to compute statistics.
        Raises ValueError if the aligned data is shorter than this.
    rolling_window : int
        Look-back window (in bars) for rolling direction-agreement series.
    """

    min_common_bars: int = 30
    rolling_window: int = 20


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PairwiseConcordance:
    """Concordance statistics for a single asset pair.

    Attributes
    ----------
    asset_a : str
        Symbol of the first asset.
    asset_b : str
        Symbol of the second asset.
    direction_agreement : float
        Fraction of bars where both assets share the same bull/bear
        direction.  Bull = regime score > 0; bear = regime score <= 0.
    score_correlation : float
        Pearson correlation of the two assets' regime scores over the
        common aligned index.
    rolling_agreement : pd.Series
        Rolling ``direction_agreement`` computed with
        ``ConcordanceConfig.rolling_window``.  The first
        ``rolling_window - 1`` values are NaN.  Index matches the
        aligned scores index.
    lead_lag_days : int
        Lag (in bars) at which the cross-correlation between the two
        direction series is maximised.  Searched over lags −10 to +10.
        Positive lag means ``asset_a`` leads ``asset_b``.
    """

    asset_a: str
    asset_b: str
    direction_agreement: float
    score_correlation: float
    rolling_agreement: pd.Series
    lead_lag_days: int


@dataclass
class ConcordanceResult:
    """Output of :func:`compute_concordance`.

    Attributes
    ----------
    assets : list[str]
        Asset symbols in analysis order.
    n_common_bars : int
        Number of bars in the aligned index.
    pairwise : list[PairwiseConcordance]
        One entry per unique ordered pair (asset_a, asset_b).
    global_agreement : float
        Mean ``direction_agreement`` across all pairs.
    regime_sync_matrix : pd.DataFrame
        Square DataFrame (assets × assets) whose (i, j) entry is
        ``direction_agreement`` for the pair (asset_i, asset_j).
        Diagonal entries are 1.0.
    """

    assets: list[str]
    n_common_bars: int
    pairwise: list[PairwiseConcordance]
    global_agreement: float
    regime_sync_matrix: pd.DataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _direction_series(scores: pd.Series) -> pd.Series:
    """Convert regime scores to binary bull (True) / bear (False) series.

    Bull = score > 0; bear = score <= 0.
    """
    return scores > 0


def _direction_agreement(dir_a: pd.Series, dir_b: pd.Series) -> float:
    """Fraction of bars where dir_a and dir_b agree (both bull or both bear).

    Parameters
    ----------
    dir_a, dir_b : pd.Series of bool
        Must have the same index.

    Returns
    -------
    float
        Value in [0, 1].
    """
    n = len(dir_a)
    if n == 0:
        return float("nan")
    return float((dir_a == dir_b).sum()) / n


def _rolling_direction_agreement(
    dir_a: pd.Series,
    dir_b: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling fraction of bars with matching direction.

    Parameters
    ----------
    dir_a, dir_b : pd.Series of bool
        Same index and length.
    window : int
        Look-back window in bars.

    Returns
    -------
    pd.Series
        Float values in [0, 1]; first ``window - 1`` entries are NaN.
    """
    agree = (dir_a == dir_b).astype(float)
    return agree.rolling(window=window, min_periods=window).mean()


def _lead_lag(dir_a: pd.Series, dir_b: pd.Series, max_lag: int = 10) -> int:
    """Lag at which cross-correlation between direction series is maximised.

    Parameters
    ----------
    dir_a, dir_b : pd.Series of bool
        Same index.
    max_lag : int
        Search range: lags from ``-max_lag`` to ``+max_lag`` inclusive.

    Returns
    -------
    int
        Lag in [-max_lag, max_lag].  Positive means asset_a leads asset_b.

    Notes
    -----
    For lag L > 0: correlate dir_a[L:] with dir_b[:-L]  (a leads b).
    For lag L < 0: correlate dir_a[:L] with dir_b[-L:]   (b leads a).
    For lag 0: correlate dir_a with dir_b.
    """
    a = dir_a.astype(float).values
    b = dir_b.astype(float).values
    T = len(a)

    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            seg_a = a[lag:]
            seg_b = b[:T - lag]
        elif lag < 0:
            seg_a = a[:T + lag]
            seg_b = b[-lag:]
        else:
            seg_a = a
            seg_b = b

        if len(seg_a) < 2:
            continue

        # Pearson correlation of binary series.
        std_a = seg_a.std()
        std_b = seg_b.std()
        if std_a < 1e-12 or std_b < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(seg_a, seg_b)[0, 1])

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_concordance(
    aligned_scores: pd.DataFrame,
    config: ConcordanceConfig | None = None,
) -> ConcordanceResult:
    """Compute regime concordance statistics from aligned regime scores.

    Parameters
    ----------
    aligned_scores : pd.DataFrame
        Columns = asset symbols, index = DatetimeIndex.
        Values = regime scores in [-1, 1] (from
        ``CrossAssetResult.aligned_scores``).
    config : ConcordanceConfig, optional
        Defaults to ``ConcordanceConfig()``.

    Returns
    -------
    ConcordanceResult

    Raises
    ------
    ValueError
        If fewer than 2 columns are present.
    ValueError
        If the number of rows is below ``config.min_common_bars``.

    Notes
    -----
    Bull direction = regime score > 0.  Bear direction = regime score <= 0.

    Direction agreement for a pair = fraction of common bars where both
    assets have the same bull/bear direction.

    Lead-lag: for each pair, cross-correlation is evaluated at lags
    −10 to +10 bars; the lag of the peak is returned.  Positive lag means
    ``asset_a`` leads ``asset_b``.
    """
    if config is None:
        config = ConcordanceConfig()

    n_assets = aligned_scores.shape[1]
    if n_assets < 2:
        raise ValueError(
            f"compute_concordance requires at least 2 asset columns, "
            f"got {n_assets}."
        )

    n_bars = len(aligned_scores)
    if n_bars < config.min_common_bars:
        raise ValueError(
            f"compute_concordance requires at least {config.min_common_bars} "
            f"common bars; got {n_bars}."
        )

    symbols = list(aligned_scores.columns)
    logger.info(
        "Computing regime concordance: %d assets, %d bars", n_assets, n_bars
    )

    # Pre-compute direction series for each asset.
    directions: dict[str, pd.Series] = {
        sym: _direction_series(aligned_scores[sym]) for sym in symbols
    }

    # ── Pairwise concordance ───────────────────────────────────────────────
    pairwise: list[PairwiseConcordance] = []
    sync_values: dict[tuple[str, str], float] = {}

    for sym_a, sym_b in combinations(symbols, 2):
        dir_a = directions[sym_a]
        dir_b = directions[sym_b]

        agreement = _direction_agreement(dir_a, dir_b)
        rolling_agr = _rolling_direction_agreement(
            dir_a, dir_b, config.rolling_window
        )
        lead_lag = _lead_lag(dir_a, dir_b)

        scores_a = aligned_scores[sym_a].values.astype(float)
        scores_b = aligned_scores[sym_b].values.astype(float)
        std_a = scores_a.std()
        std_b = scores_b.std()
        if std_a < 1e-12 or std_b < 1e-12:
            score_corr = float("nan")
        else:
            score_corr = float(np.corrcoef(scores_a, scores_b)[0, 1])

        pairwise.append(
            PairwiseConcordance(
                asset_a=sym_a,
                asset_b=sym_b,
                direction_agreement=agreement,
                score_correlation=score_corr,
                rolling_agreement=rolling_agr,
                lead_lag_days=lead_lag,
            )
        )

        sync_values[(sym_a, sym_b)] = agreement
        sync_values[(sym_b, sym_a)] = agreement  # symmetric

    # ── Sync matrix ──────────────────────────────────────────────────────────
    sync_matrix = pd.DataFrame(
        {
            sym_col: [
                1.0 if sym_row == sym_col else sync_values.get((sym_row, sym_col), float("nan"))
                for sym_row in symbols
            ]
            for sym_col in symbols
        },
        index=symbols,
    )

    # ── Global agreement ─────────────────────────────────────────────────────
    pair_agreements = [p.direction_agreement for p in pairwise]
    global_agreement = float(np.nanmean(pair_agreements)) if pair_agreements else float("nan")

    logger.info(
        "Concordance computed: global_agreement=%.3f, %d pairs",
        global_agreement,
        len(pairwise),
    )

    return ConcordanceResult(
        assets=symbols,
        n_common_bars=n_bars,
        pairwise=pairwise,
        global_agreement=global_agreement,
        regime_sync_matrix=sync_matrix,
    )


def concordance_heatmap_data(result: ConcordanceResult) -> pd.DataFrame:
    """Return the regime_sync_matrix ready for plotting.

    Parameters
    ----------
    result : ConcordanceResult

    Returns
    -------
    pd.DataFrame
        The ``regime_sync_matrix`` from ``result``.  Values are fractions
        in [0, 1]; diagonal is 1.0.
    """
    return result.regime_sync_matrix.copy()


def rolling_concordance_series(
    aligned_scores: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Compute rolling direction-agreement for every asset pair.

    Parameters
    ----------
    aligned_scores : pd.DataFrame
        Columns = asset symbols, index = DatetimeIndex.
        Values = regime scores in [-1, 1].
    window : int
        Look-back window in bars.

    Returns
    -------
    pd.DataFrame
        Columns are named ``"{asset_a}__{asset_b}"`` for each unique pair.
        Values are rolling fractions in [0, 1]; first ``window - 1`` entries
        in each column are NaN.

    Notes
    -----
    The number of columns equals ``n_assets * (n_assets - 1) // 2``.
    """
    symbols = list(aligned_scores.columns)
    result_cols: dict[str, pd.Series] = {}

    for sym_a, sym_b in combinations(symbols, 2):
        dir_a = _direction_series(aligned_scores[sym_a])
        dir_b = _direction_series(aligned_scores[sym_b])
        col_name = f"{sym_a}__{sym_b}"
        result_cols[col_name] = _rolling_direction_agreement(dir_a, dir_b, window)

    return pd.DataFrame(result_cols, index=aligned_scores.index)
