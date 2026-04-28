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
from rde.analysis.risk_metrics import (
    RegimeRiskConfig,
    RegimeRiskResult,
    compute_regime_risk,
    compute_regime_risk_weighted,
    regime_risk_divergence,
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
    # risk metrics
    "RegimeRiskConfig",
    "RegimeRiskResult",
    "compute_regime_risk",
    "compute_regime_risk_weighted",
    "regime_risk_divergence",
]
