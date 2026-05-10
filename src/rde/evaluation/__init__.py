"""Regime evaluation: persistence, stability, transition analysis, walk-forward harness."""

from rde.evaluation.baselines import (
    BaselineResult,
    buy_and_hold,
    compare_model_to_baselines,
    multi_asset_risk_parity,
    naive_momentum,
    naive_vol_regime,
    risk_parity_baseline,
    run_all_baselines,
    two_state_hmm_baseline,
    vol_targeted_buy_and_hold,
)
from rde.evaluation.feature_importance import (
    FeatureImportanceResult,
    FoldImportance,
    run_permutation_importance,
)
from rde.evaluation.honest_tearsheet import (
    HonestTearsheet,
    compute_honest_tearsheet,
    write_honest_tearsheet,
)
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
from rde.evaluation.skeptics import (
    AblationResult,
    CostSweepResult,
    NullTestResult,
    PeriodRobustnessResult,
    SkepticsReport,
    run_cost_sensitivity,
    run_feature_ablation,
    run_full_skeptics_kit,
    run_period_robustness,
    run_random_baseline,
    run_shuffle_test,
    run_slippage_stress,
    write_skeptics_report,
)
from rde.evaluation.stability import stability_across_restarts
from rde.evaluation.transition import stationary_distribution, transition_entropy
from rde.evaluation.vol_forecasting import (
    VolForecastResult,
    compare_vol_forecasters,
    run_vol_forecast_cv,
)
from rde.evaluation.walk_forward import WalkForwardHarness

__all__ = [
    # Phase 37.2 — Skeptic's kit
    "NullTestResult",
    "AblationResult",
    "PeriodRobustnessResult",
    "CostSweepResult",
    "SkepticsReport",
    "run_random_baseline",
    "run_shuffle_test",
    "run_feature_ablation",
    "run_period_robustness",
    "run_cost_sensitivity",
    "run_slippage_stress",
    "run_full_skeptics_kit",
    "write_skeptics_report",
    # Phase 37.3 — Baselines
    "BaselineResult",
    "buy_and_hold",
    "vol_targeted_buy_and_hold",
    "naive_momentum",
    "naive_vol_regime",
    "two_state_hmm_baseline",
    "risk_parity_baseline",
    "multi_asset_risk_parity",
    "run_all_baselines",
    "compare_model_to_baselines",
    # Phase 37.4 — Feature importance
    "FoldImportance",
    "FeatureImportanceResult",
    "run_permutation_importance",
    # Phase 37.5 — Honest tearsheet
    "HonestTearsheet",
    "compute_honest_tearsheet",
    "write_honest_tearsheet",
    # Phase 37.1 — Purged CV
    "FoldResult",
    "purged_k_fold_splits",
    "combinatorial_purged_splits",
    "run_purged_cv",
    "run_combinatorial_purged_cv",
    # Pre-37 evaluation
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
    # Phase 46 — Vol forecasting quality
    "VolForecastResult",
    "run_vol_forecast_cv",
    "compare_vol_forecasters",
]
