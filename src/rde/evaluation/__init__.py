"""Regime evaluation: persistence, stability, transition analysis, walk-forward harness."""

from rde.evaluation.persistence import empirical_dwell_times, expected_dwell_times
from rde.evaluation.purged_cv import (
    FoldResult,
    combinatorial_purged_splits,
    purged_k_fold_splits,
    run_combinatorial_purged_cv,
    run_purged_cv,
)
from rde.evaluation.regime_analytics import (
    RegimeStats,
    compute_regime_stats,
    regime_stats_to_dataframe,
    regime_transition_table,
    transition_forecast,
)
from rde.evaluation.stability import stability_across_restarts
from rde.evaluation.transition import stationary_distribution, transition_entropy
from rde.evaluation.walk_forward import WalkForwardHarness

__all__ = [
    "empirical_dwell_times",
    "expected_dwell_times",
    "stability_across_restarts",
    "stationary_distribution",
    "transition_entropy",
    "WalkForwardHarness",
    "RegimeStats",
    "compute_regime_stats",
    "regime_stats_to_dataframe",
    "regime_transition_table",
    "transition_forecast",
    "FoldResult",
    "purged_k_fold_splits",
    "combinatorial_purged_splits",
    "run_purged_cv",
    "run_combinatorial_purged_cv",
]
