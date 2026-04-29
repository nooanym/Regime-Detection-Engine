"""Cross-asset regime correlation analysis.

For each pair of assets, this module computes:

1. Unconditional return correlation.
2. Return correlation conditioned on the joint regime direction
   (bull/bull, bull/bear, bear/bull, bear/bear), where bull = regime
   score > 0 and bear = regime score ≤ 0.
3. Regime-score correlation: how synchronised the regime sentiment is
   across assets (a continuous analogue of the binary alignment).
4. Regime co-occurrence: for every pair of assets, a 2×2 matrix counting
   how many hours each joint direction pair occurred.

A "regime score" for each bar is derived from the asset's saved
regime_analytics.parquet: states are ranked by annualised Sharpe ratio
and mapped to [-1, 1]. This works without requiring --signal to have
been run, as long as regimes.parquet and regime_analytics.parquet exist.

Output tables
-------------
``return_corr``          pd.DataFrame  — unconditional pairwise return correlation
``score_corr``           pd.DataFrame  — pairwise regime-score correlation
``cond_return_corr``     dict[str, pd.DataFrame]  — keyed "bull_bull" etc.
``co_occurrence``        dict[tuple, pd.DataFrame] — counts per asset pair
``aligned_returns``      pd.DataFrame  — hourly returns on the common index
``aligned_scores``       pd.DataFrame  — regime scores on the common index
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DIRECTIONS = ("bull_bull", "bull_bear", "bear_bull", "bear_bear")


@dataclass
class AssetRegimeData:
    """Per-asset data loaded from a results directory.

    Attributes
    ----------
    symbol : str
        Asset ticker.
    returns : pd.Series
        Per-bar log returns (tz-aware index).
    regime_score : pd.Series
        Continuous regime sentiment in [-1, 1] derived from Sharpe-ranked
        state scores. Higher = better expected regime.
    regime_label : pd.Series
        Human-readable label per bar.
    """

    symbol: str
    returns: pd.Series
    regime_score: pd.Series
    regime_label: pd.Series


@dataclass
class CrossAssetResult:
    """Output of :func:`compute_cross_asset`.

    Attributes
    ----------
    assets : list[str]
        Asset symbols in analysis order.
    n_common_bars : int
        Number of bars on the common aligned index.
    return_corr : pd.DataFrame
        Unconditional pairwise Pearson return correlation.
    score_corr : pd.DataFrame
        Pairwise regime-score correlation.
    cond_return_corr : dict[str, pd.DataFrame]
        Conditional return correlation for each of the four joint
        directions: ``"bull_bull"``, ``"bull_bear"``,
        ``"bear_bull"``, ``"bear_bear"``.
    co_occurrence : dict[tuple[str, str], pd.DataFrame]
        For each asset pair, a 2×2 count DataFrame (index/columns:
        ``["bull", "bear"]``) counting how many hours were spent in
        each joint direction.
    aligned_returns : pd.DataFrame
        Log returns for all assets on the common index.
    aligned_scores : pd.DataFrame
        Regime scores for all assets on the common index.
    """

    assets: list[str]
    n_common_bars: int
    return_corr: pd.DataFrame
    score_corr: pd.DataFrame
    cond_return_corr: dict[str, pd.DataFrame]
    co_occurrence: dict[tuple[str, str], pd.DataFrame]
    aligned_returns: pd.DataFrame
    aligned_scores: pd.DataFrame


def _sharpe_scores(analytics_df: pd.DataFrame) -> np.ndarray:
    """Rank states by sharpe_ann, return scores in [-1, 1].

    Parameters
    ----------
    analytics_df : pd.DataFrame
        Output of ``regime_stats_to_dataframe``. Must have a ``state``
        column and a ``sharpe_ann`` column.

    Returns
    -------
    np.ndarray
        Shape (n_states,). Index i gives the score for state i.
    """
    n = len(analytics_df)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])

    sharpe = (analytics_df["sharpe_ann"] if "sharpe_ann" in analytics_df.columns
              else analytics_df.set_index("state")["sharpe_ann"]).fillna(0.0)
    ranks = sharpe.rank(method="average") - 1.0  # 0-based
    scores = 2.0 * ranks / (n - 1) - 1.0
    return scores.values


def load_asset_regime_data(result_dir: Path | str) -> AssetRegimeData:
    """Load regime data for a single asset from its result directory.

    Parameters
    ----------
    result_dir : Path or str
        The per-asset output directory, e.g. ``results/BTC-USD``.
        Must contain ``regimes.parquet`` and ``regime_analytics.parquet``.

    Returns
    -------
    AssetRegimeData

    Raises
    ------
    FileNotFoundError
        If either required parquet is absent.
    """
    result_dir = Path(result_dir)
    reg_path = result_dir / "regimes.parquet"
    ana_path = result_dir / "regime_analytics.parquet"

    for p in (reg_path, ana_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    regimes_df = pd.read_parquet(reg_path)
    analytics_df = pd.read_parquet(ana_path)

    if "log_return" not in regimes_df.columns:
        raise KeyError(
            f"'log_return' column missing from {reg_path}. "
            "Re-run the pipeline to regenerate this file."
        )
    if "regime" not in regimes_df.columns:
        raise KeyError(f"'regime' column missing from {reg_path}.")

    state_scores = _sharpe_scores(analytics_df)
    regime_score_vals = np.array(
        [state_scores[int(s)] if int(s) < len(state_scores) else 0.0
         for s in regimes_df["regime"]]
    )

    symbol = result_dir.name
    returns = pd.Series(
        regimes_df["log_return"].values,
        index=regimes_df.index,
        name=symbol,
    )
    regime_score = pd.Series(
        regime_score_vals,
        index=regimes_df.index,
        name=symbol,
    )
    regime_label = pd.Series(
        regimes_df.get("regime_label", regimes_df["regime"]).values,
        index=regimes_df.index,
        name=symbol,
    )

    logger.info(
        "Loaded %s: %d bars  (%s → %s)",
        symbol,
        len(regimes_df),
        regimes_df.index[0].date(),
        regimes_df.index[-1].date(),
    )
    return AssetRegimeData(
        symbol=symbol,
        returns=returns,
        regime_score=regime_score,
        regime_label=regime_label,
    )


def compute_cross_asset(
    data: list[AssetRegimeData],
    resample_freq: str | None = None,
) -> CrossAssetResult:
    """Compute cross-asset regime correlation statistics.

    Parameters
    ----------
    data : list[AssetRegimeData]
        Per-asset data from :func:`load_asset_regime_data`.
        Must contain at least 2 assets.
    resample_freq : str or None
        If given (e.g. ``"1D"``), resample all series to this frequency
        before aligning.  Returns are summed; regime scores take the last
        value in each period.  Useful when mixing crypto (24/7 hourly) with
        equities (market-hours hourly) whose bar timestamps don't line up.

    Returns
    -------
    CrossAssetResult

    Raises
    ------
    ValueError
        If fewer than 2 assets are provided, or the common index is empty.
    """
    if len(data) < 2:
        raise ValueError(f"Need at least 2 assets, got {len(data)}.")

    # ── Optional resampling ───────────────────────────────────────────────────
    if resample_freq is not None:
        resampled: list[AssetRegimeData] = []
        for d in data:
            ret_rs = d.returns.resample(resample_freq).sum()
            score_rs = d.regime_score.resample(resample_freq).last()
            label_rs = d.regime_label.resample(resample_freq).last()
            resampled.append(
                AssetRegimeData(
                    symbol=d.symbol,
                    returns=ret_rs,
                    regime_score=score_rs,
                    regime_label=label_rs,
                )
            )
        data = resampled
        logger.info("Resampled all assets to %s frequency", resample_freq)

    # ── Align on common index ─────────────────────────────────────────────────
    common_idx = data[0].returns.index
    for d in data[1:]:
        common_idx = common_idx.intersection(d.returns.index)

    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping timestamps across assets. "
            "Check that all assets use the same period and interval. "
            "For mixed markets (crypto + equities), try resample_freq='1D'."
        )

    symbols = [d.symbol for d in data]
    aligned_returns = pd.DataFrame(
        {d.symbol: d.returns.reindex(common_idx) for d in data}
    )
    aligned_scores = pd.DataFrame(
        {d.symbol: d.regime_score.reindex(common_idx) for d in data}
    )

    logger.info(
        "Cross-asset alignment: %d common bars across %s",
        len(common_idx), symbols,
    )

    # ── Unconditional correlations ────────────────────────────────────────────
    return_corr = aligned_returns.corr()
    score_corr = aligned_scores.corr()

    # ── Conditional return correlations ──────────────────────────────────────
    bull = aligned_scores > 0  # True = bull regime

    cond_return_corr: dict[str, pd.DataFrame] = {}
    for direction in _DIRECTIONS:
        d_a, d_b = direction.split("_")
        a_bull = d_a == "bull"
        b_bull = d_b == "bull"

        cond_corrs: dict[tuple[str, str], float] = {}
        for sym_a, sym_b in combinations(symbols, 2):
            mask = (bull[sym_a] == a_bull) & (bull[sym_b] == b_bull)
            subset = aligned_returns.loc[mask, [sym_a, sym_b]].dropna()
            if len(subset) >= 10:
                c = subset[sym_a].corr(subset[sym_b])
            else:
                c = float("nan")
            cond_corrs[(sym_a, sym_b)] = c
            cond_corrs[(sym_b, sym_a)] = c

        mat = pd.DataFrame(
            {
                sym: [
                    cond_corrs.get((sym, other), 1.0 if sym == other else float("nan"))
                    for other in symbols
                ]
                for sym in symbols
            },
            index=symbols,
        )
        cond_return_corr[direction] = mat

    # ── Co-occurrence counts ──────────────────────────────────────────────────
    co_occurrence: dict[tuple[str, str], pd.DataFrame] = {}
    for sym_a, sym_b in combinations(symbols, 2):
        counts = pd.DataFrame(
            0,
            index=pd.Index(["bull", "bear"], name=sym_a),
            columns=pd.Index(["bull", "bear"], name=sym_b),
        )
        for a_dir in ("bull", "bear"):
            for b_dir in ("bull", "bear"):
                a_mask = bull[sym_a] if a_dir == "bull" else ~bull[sym_a]
                b_mask = bull[sym_b] if b_dir == "bull" else ~bull[sym_b]
                counts.loc[a_dir, b_dir] = int((a_mask & b_mask).sum())
        co_occurrence[(sym_a, sym_b)] = counts

    return CrossAssetResult(
        assets=symbols,
        n_common_bars=len(common_idx),
        return_corr=return_corr,
        score_corr=score_corr,
        cond_return_corr=cond_return_corr,
        co_occurrence=co_occurrence,
        aligned_returns=aligned_returns,
        aligned_scores=aligned_scores,
    )
