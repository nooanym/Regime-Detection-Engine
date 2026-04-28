"""Cross-asset and portfolio-level analysis utilities."""

from rde.analysis.cross_asset import (
    AssetRegimeData,
    CrossAssetResult,
    compute_cross_asset,
    load_asset_regime_data,
)
from rde.analysis.portfolio import (
    AllocationConfig,
    KellyConfig,
    VolTargetConfig,
    kelly_weights,
    portfolio_returns,
    score_proportional_weights,
    vol_target_weights,
)
from rde.analysis.regime_var import (
    RegimeVARConfig,
    RegimeVARResult,
    compute_irf,
    fit_regime_var,
    granger_causality_table,
)
from rde.analysis.drawdown_control import (
    DrawdownControlConfig,
    RegimePositionLimitConfig,
    apply_drawdown_control,
    apply_regime_position_limits,
    drawdown_scale_factor,
    regime_drawdown_budget,
    regime_position_limits,
    running_drawdown,
)
from rde.analysis.information_geometry import (
    DistinguishabilityResult,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    compute_distinguishability,
    js_divergence_gaussians,
    kl_divergence_gaussians,
    mahalanobis_distance,
    markov_entropy_rate,
    regime_entropy,
)
from rde.analysis.risk_metrics import (
    RegimeRiskConfig,
    RegimeRiskResult,
    compute_regime_risk,
    compute_regime_risk_weighted,
    regime_risk_divergence,
)
from rde.analysis.cointegration import (
    RegimeCointegrationResult,
    RegimeSpreadParams,
    SpreadConfig,
    fit_regime_spreads,
    half_life_ornstein_uhlenbeck,
    regime_half_lives,
    regime_spread_zscore,
    spread_zscore,
    weighted_ols,
)
from rde.analysis.factor_analysis import (
    FactorConfig,
    RegimeFactorResult,
    RegimeFactors,
    factor_alignment,
    factor_return_decomposition,
    fit_regime_factors,
    project_to_factors,
    rolling_factor_exposure,
)
from rde.analysis.execution import (
    ExecutionConfig,
    ImpactModelConfig,
    OrderSchedule,
    RegimeImpactParams,
    estimate_regime_impact,
    expected_slippage,
    optimal_order_schedule,
    slippage_attribution,
    twap_schedule,
    vwap_schedule,
)

__all__ = [
    # cross-asset
    "AssetRegimeData",
    "CrossAssetResult",
    "compute_cross_asset",
    "load_asset_regime_data",
    # portfolio
    "AllocationConfig",
    "KellyConfig",
    "VolTargetConfig",
    "kelly_weights",
    "portfolio_returns",
    "score_proportional_weights",
    "vol_target_weights",
    # regime VAR
    "RegimeVARConfig",
    "RegimeVARResult",
    "fit_regime_var",
    "compute_irf",
    "granger_causality_table",
    # drawdown control
    "DrawdownControlConfig",
    "RegimePositionLimitConfig",
    "running_drawdown",
    "drawdown_scale_factor",
    "apply_drawdown_control",
    "regime_position_limits",
    "apply_regime_position_limits",
    "regime_drawdown_budget",
    # information geometry
    "DistinguishabilityResult",
    "kl_divergence_gaussians",
    "js_divergence_gaussians",
    "bhattacharyya_distance",
    "bhattacharyya_coefficient",
    "mahalanobis_distance",
    "markov_entropy_rate",
    "regime_entropy",
    "compute_distinguishability",
    # risk metrics
    "RegimeRiskConfig",
    "RegimeRiskResult",
    "compute_regime_risk",
    "compute_regime_risk_weighted",
    "regime_risk_divergence",
    # cointegration / spread
    "SpreadConfig",
    "RegimeSpreadParams",
    "RegimeCointegrationResult",
    "weighted_ols",
    "half_life_ornstein_uhlenbeck",
    "fit_regime_spreads",
    "spread_zscore",
    "regime_spread_zscore",
    "regime_half_lives",
    # factor analysis
    "FactorConfig",
    "RegimeFactors",
    "RegimeFactorResult",
    "fit_regime_factors",
    "factor_alignment",
    "rolling_factor_exposure",
    "project_to_factors",
    "factor_return_decomposition",
    # execution
    "ImpactModelConfig",
    "ExecutionConfig",
    "RegimeImpactParams",
    "OrderSchedule",
    "estimate_regime_impact",
    "expected_slippage",
    "optimal_order_schedule",
    "twap_schedule",
    "vwap_schedule",
    "slippage_attribution",
]
