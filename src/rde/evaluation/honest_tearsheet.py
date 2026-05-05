"""Honest strategy tearsheet aggregating all Phase 37 validation (Phase 37.5).

Assembles the purged CV fold results (37.1), skeptic's kit (37.2), baseline
comparison (37.3), and permutation feature importance (37.4) into a single
:class:`HonestTearsheet` struct and writes a human-readable Markdown report.

This is the decision gate before Phase 38. Read the tearsheet before
proceeding with deployment infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from rde.evaluation.baselines import BaselineResult, compare_model_to_baselines
from rde.evaluation.feature_importance import FeatureImportanceResult
from rde.evaluation.purged_cv import FoldResult
from rde.evaluation.skeptics import SkepticsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class HonestTearsheet:
    """Aggregated honest validation results for one asset and model configuration.

    Attributes
    ----------
    asset : str
    date_generated : str
        ISO-8601 UTC timestamp of report generation.
    n_cv_folds : int
    sharpe_mean : float
    sharpe_std : float
    sharpe_worst_5pct : float
        5th percentile of fold Sharpe distribution.
    mdd_mean : float
    mdd_std : float
    mdd_worst_5pct : float
        95th percentile (worst-case) of fold max-drawdown distribution.
    cost_break_even_bps : float
        Transaction cost level at which Sharpe crosses zero.
    slippage_break_even_bps : float
        Slippage level at which Sharpe crosses zero.
    baseline_comparison : pd.DataFrame
        Columns: name, baseline_sharpe, model_sharpe, beats, margin.
    beats_all_baselines : bool
        ``True`` if the model beats every baseline.
    random_baseline_passed : bool
    shuffle_test_passed : bool
    period_robustness_stable : bool
    stable_features : list[str]
        Features with ``positive_fold_fraction >= 0.7``.
    unstable_features : list[str]
        Features with ``positive_fold_fraction < 0.7``.
    capacity_estimate_usd : float
        Rough USD capacity estimate (where market impact ≈ alpha).
    capacity_assumptions : str
        Human-readable assumptions used in the capacity estimate.
    failure_modes : list[str]
        Qualitative failure modes identified from the validation results.
    """

    asset: str
    date_generated: str
    n_cv_folds: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_worst_5pct: float
    mdd_mean: float
    mdd_std: float
    mdd_worst_5pct: float
    cost_break_even_bps: float
    slippage_break_even_bps: float
    baseline_comparison: pd.DataFrame
    beats_all_baselines: bool
    random_baseline_passed: bool
    shuffle_test_passed: bool
    period_robustness_stable: bool
    stable_features: list[str]
    unstable_features: list[str]
    capacity_estimate_usd: float
    capacity_assumptions: str
    failure_modes: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_capacity(
    ann_return: float,
    n_trades_per_year: float,
    avg_trade_size_frac: float = 1.0,
    daily_volume_usd: float = 20_000_000_000.0,
    impact_coeff: float = 0.001,
) -> float:
    """Estimate AUM capacity using the square-root impact model.

    Break-even occurs when impact per trade equals alpha per trade:

        alpha_per_trade = ann_return / n_trades_per_year
        impact = impact_coeff * sqrt(trade_usd / daily_volume_usd)
        => trade_usd = (alpha_per_trade / impact_coeff)^2 * daily_volume_usd

    The total AUM capacity is ``trade_usd / avg_trade_size_frac``.

    Parameters
    ----------
    ann_return : float
        Model annualised return.
    n_trades_per_year : float
        Average trades per year from CV folds.
    avg_trade_size_frac : float
        Average position fraction (default 1.0 = full position).
    daily_volume_usd : float
        Asset daily trading volume in USD (default $20B for BTC).
    impact_coeff : float
        Square-root impact coefficient (default 0.001 = 0.1%).

    Returns
    -------
    float
        Capacity in USD. Returns ``inf`` if n_trades_per_year == 0.
    """
    if n_trades_per_year <= 0 or not np.isfinite(ann_return):
        return float("inf")
    alpha_per_trade = ann_return / n_trades_per_year
    if alpha_per_trade <= 0:
        return 0.0
    trade_usd = (alpha_per_trade / (impact_coeff + 1e-15)) ** 2 * daily_volume_usd
    capacity = trade_usd / max(avg_trade_size_frac, 1e-9)
    return float(capacity)


def _identify_failure_modes(
    fold_results: list[FoldResult],
    skeptics: SkepticsReport,
    importance: FeatureImportanceResult,
    baseline_comp: pd.DataFrame,
) -> list[str]:
    """Generate a list of qualitative failure modes from validation results."""
    modes: list[str] = []

    # Regime signal may not add value beyond noise
    if not skeptics.random_baseline.passed:
        modes.append(
            f"Random regime baseline has comparable Sharpe (margin "
            f"{skeptics.random_baseline.margin:.2f} < 0.3). "
            "The HMM's temporal structure may not add value beyond marginal regime frequencies."
        )

    if not skeptics.shuffle_test.passed:
        modes.append(
            f"Shuffle test failed (margin {skeptics.shuffle_test.margin:.2f} < 0.3). "
            "The ordering of regime labels appears unimportant — the strategy may be "
            "coincidentally profiting from the marginal distribution."
        )

    # Period instability
    if not skeptics.period_robustness.is_stable:
        modes.append(
            f"Period robustness: ARI mean={skeptics.period_robustness.ari_mean:.2f} "
            f"(threshold 0.70). Regime structure shifts across time periods — "
            "the model may not generalise to unseen regimes."
        )

    # Tail risk: worst-5% fold
    sharpes = [fr.sharpe for fr in fold_results if np.isfinite(fr.sharpe)]
    if sharpes:
        worst_5 = float(np.percentile(sharpes, 5))
        if worst_5 < -0.5:
            modes.append(
                f"Worst-5% fold Sharpe is {worst_5:.2f}. Significant tail risk in "
                "adverse market periods."
            )

    # Baseline underperformance
    if not baseline_comp.empty:
        beaten = baseline_comp["beats"].sum()
        total = len(baseline_comp)
        if beaten < total:
            failed = baseline_comp[~baseline_comp["beats"]]["name"].tolist()
            modes.append(
                f"Model does not beat all baselines: fails vs {', '.join(failed)}. "
                "Simpler strategies may achieve equivalent or better risk-adjusted returns."
            )

    # Unstable features
    unstable = [f for f, stable in importance.is_stable.items() if not stable]
    if unstable:
        modes.append(
            f"Features with inconsistent importance across folds (positive-fold fraction < 0.7): "
            f"{', '.join(unstable)}. These features may contribute only in specific regimes."
        )

    # High drawdown
    mdds = [fr.max_drawdown for fr in fold_results if np.isfinite(fr.max_drawdown)]
    if mdds:
        worst_mdd = float(np.percentile(mdds, 95))
        if worst_mdd > 0.20:
            modes.append(
                f"95th-percentile fold max-drawdown is {worst_mdd:.1%}. "
                "Position sizing and stop-loss rules should be reviewed before live trading."
            )

    # Cost sensitivity
    if np.isfinite(skeptics.cost_sensitivity.break_even_bps):
        be = skeptics.cost_sensitivity.break_even_bps
        if be < 5.0:
            modes.append(
                f"Cost break-even is only {be:.1f} bps/side — the strategy is fragile "
                "to transaction costs. Execution quality is critical."
            )

    if not modes:
        modes.append(
            "No critical failures identified. Review all sections carefully before "
            "deployment — this list is automatically generated and may miss qualitative risks."
        )

    return modes


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_honest_tearsheet(
    fold_results: list[FoldResult],
    baseline_results: dict[str, BaselineResult],
    skeptics_report: SkepticsReport,
    importance_result: FeatureImportanceResult,
    daily_volume_usd: float = 20_000_000_000.0,
    asset: str = "unknown",
) -> HonestTearsheet:
    """Aggregate all Phase 37 results into a single :class:`HonestTearsheet`.

    Parameters
    ----------
    fold_results : list[FoldResult]
        From :func:`rde.evaluation.purged_cv.run_purged_cv`.
    baseline_results : dict[str, BaselineResult]
        From :func:`rde.evaluation.baselines.run_all_baselines`.
    skeptics_report : SkepticsReport
        From :func:`rde.evaluation.skeptics.run_full_skeptics_kit`.
    importance_result : FeatureImportanceResult
        From :func:`rde.evaluation.feature_importance.run_permutation_importance`.
    daily_volume_usd : float
        Asset daily trading volume for capacity estimate (default $20B for BTC).
    asset : str

    Returns
    -------
    HonestTearsheet
    """
    sharpes = [fr.sharpe for fr in fold_results if np.isfinite(fr.sharpe)]
    mdds = [fr.max_drawdown for fr in fold_results if np.isfinite(fr.max_drawdown)]

    sharpe_mean = float(np.mean(sharpes)) if sharpes else float("nan")
    sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    sharpe_worst_5pct = float(np.percentile(sharpes, 5)) if sharpes else float("nan")
    mdd_mean = float(np.mean(mdds)) if mdds else float("nan")
    mdd_std = float(np.std(mdds, ddof=1)) if len(mdds) > 1 else 0.0
    mdd_worst_5pct = float(np.percentile(mdds, 95)) if mdds else float("nan")

    baseline_comp = compare_model_to_baselines(sharpe_mean, baseline_results)
    beats_all = bool(baseline_comp["beats"].all()) if not baseline_comp.empty else False

    stable = [f for f, s in importance_result.is_stable.items() if s]
    unstable = [f for f, s in importance_result.is_stable.items() if not s]

    # Capacity estimate: use mean annual return and mean trades/year from folds
    ann_returns = [fr.ann_return for fr in fold_results if np.isfinite(fr.ann_return)]
    ann_ret_mean = float(np.mean(ann_returns)) if ann_returns else 0.0
    n_trades_vals = [fr.n_trades for fr in fold_results]
    # Folds are typically ~500 bars = ~3 weeks hourly; annualise trade count
    ann_factor = 8760
    test_bars_approx = float(np.mean([fr.n_test for fr in fold_results])) if fold_results else 500.0
    trades_per_year = float(np.mean(n_trades_vals)) * (ann_factor / max(test_bars_approx, 1.0))

    capacity = _estimate_capacity(
        ann_return=ann_ret_mean,
        n_trades_per_year=trades_per_year,
        daily_volume_usd=daily_volume_usd,
    )
    cap_assumptions = (
        f"ann_return={ann_ret_mean:.1%}, "
        f"trades_per_year≈{trades_per_year:.0f}, "
        f"daily_volume_usd=${daily_volume_usd / 1e9:.0f}B, "
        f"impact_coeff=0.001 (0.1% sqrt-impact)"
    )

    failure_modes = _identify_failure_modes(fold_results, skeptics_report, importance_result, baseline_comp)

    return HonestTearsheet(
        asset=asset,
        date_generated=datetime.now(tz=timezone.utc).isoformat(),
        n_cv_folds=len(fold_results),
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        sharpe_worst_5pct=sharpe_worst_5pct,
        mdd_mean=mdd_mean,
        mdd_std=mdd_std,
        mdd_worst_5pct=mdd_worst_5pct,
        cost_break_even_bps=skeptics_report.cost_sensitivity.break_even_bps,
        slippage_break_even_bps=skeptics_report.slippage_stress.break_even_bps,
        baseline_comparison=baseline_comp,
        beats_all_baselines=beats_all,
        random_baseline_passed=skeptics_report.random_baseline.passed,
        shuffle_test_passed=skeptics_report.shuffle_test.passed,
        period_robustness_stable=skeptics_report.period_robustness.is_stable,
        stable_features=stable,
        unstable_features=unstable,
        capacity_estimate_usd=capacity,
        capacity_assumptions=cap_assumptions,
        failure_modes=failure_modes,
    )


def write_honest_tearsheet(
    tearsheet: HonestTearsheet,
    fold_results: list[FoldResult],
    baseline_results: dict[str, BaselineResult],
    skeptics_report: SkepticsReport,
    importance_result: FeatureImportanceResult,
    results_dir: Path,
) -> Path:
    """Write the honest tearsheet as a Markdown file.

    Sections:
    1. Mean and std of Sharpe across CV folds
    2. Distribution of max drawdowns across folds
    3. Worst 5% fold performance
    4. Cost break-even level
    5. Capacity estimate (with assumptions)
    6. What the model does NOT do well — failure modes
    7. Baseline comparison table
    8. Feature importance summary
    9. Skeptic's kit pass/fail table

    Parameters
    ----------
    tearsheet : HonestTearsheet
    fold_results : list[FoldResult]
    baseline_results : dict[str, BaselineResult]
    skeptics_report : SkepticsReport
    importance_result : FeatureImportanceResult
    results_dir : Path

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "honest_tearsheet.md"

    def _fmt(v: float, fmt: str = ".3f") -> str:
        if not np.isfinite(v):
            return "N/A"
        return format(v, fmt)

    def _tick(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines: list[str] = [
        f"# Honest Strategy Tearsheet — {tearsheet.asset}",
        "",
        f"Generated: {tearsheet.date_generated}",
        f"CV folds: {tearsheet.n_cv_folds}",
        "",
        "---",
        "",
        "## 1. Sharpe Distribution Across CV Folds",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Sharpe | {_fmt(tearsheet.sharpe_mean)} |",
        f"| Std Sharpe | {_fmt(tearsheet.sharpe_std)} |",
        f"| Worst 5% fold Sharpe | {_fmt(tearsheet.sharpe_worst_5pct)} |",
        "",
    ]

    # Per-fold Sharpe table
    if fold_results:
        lines += [
            "<details><summary>Per-fold Sharpe</summary>",
            "",
            "| Fold | Test Start | Sharpe | Calmar |",
            "|------|------------|--------|--------|",
        ]
        for fr in fold_results:
            lines.append(
                f"| {fr.fold_id} | {fr.test_start.date()} | "
                f"{_fmt(fr.sharpe)} | {_fmt(fr.calmar)} |"
            )
        lines += ["", "</details>", ""]

    lines += [
        "## 2. Max Drawdown Distribution",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean MDD | {_fmt(tearsheet.mdd_mean, '.1%')} |",
        f"| Std MDD | {_fmt(tearsheet.mdd_std, '.1%')} |",
        f"| Worst 5% fold MDD | {_fmt(tearsheet.mdd_worst_5pct, '.1%')} |",
        "",
        "## 3. Worst 5% Fold Performance",
        "",
        f"- Worst-5% Sharpe: **{_fmt(tearsheet.sharpe_worst_5pct)}**",
        f"- Worst-5% MDD: **{_fmt(tearsheet.mdd_worst_5pct, '.1%')}**",
        "",
        "## 4. Cost Break-Even",
        "",
    ]
    be_cost = tearsheet.cost_break_even_bps
    be_slip = tearsheet.slippage_break_even_bps
    be_cost_str = f"{be_cost:.1f}" if np.isfinite(be_cost) else "∞ (profitable across full sweep)"
    be_slip_str = f"{be_slip:.1f}" if np.isfinite(be_slip) else "∞ (profitable across full sweep)"
    lines += [
        f"- Transaction cost break-even: **{be_cost_str} bps/side**",
        f"- Slippage break-even: **{be_slip_str} bps**",
        "",
        "| Cost (bps) | Sharpe |",
        "|------------|--------|",
    ]
    for bps, sh in zip(
        skeptics_report.cost_sensitivity.cost_bps,
        skeptics_report.cost_sensitivity.sharpes,
    ):
        lines.append(f"| {bps:.0f} | {_fmt(sh)} |")

    lines += [
        "",
        "## 5. Capacity Estimate",
        "",
        f"**Estimated capacity: ${tearsheet.capacity_estimate_usd:,.0f}**",
        "",
        f"Assumptions: {tearsheet.capacity_assumptions}",
        "",
        "Methodology: square-root market impact model. Break-even where "
        "impact per trade equals alpha per trade.",
        "",
        "## 6. What the Model Does NOT Do Well — Failure Modes",
        "",
    ]
    for i, mode in enumerate(tearsheet.failure_modes, 1):
        lines.append(f"{i}. {mode}")
    lines.append("")

    lines += [
        "## 7. Baseline Comparison",
        "",
        f"Model mean Sharpe (CV): **{_fmt(tearsheet.sharpe_mean)}**",
        "",
        "| Baseline | Baseline Sharpe | Model Sharpe | Beats | Margin |",
        "|----------|-----------------|--------------|-------|--------|",
    ]
    for _, row in tearsheet.baseline_comparison.iterrows():
        beats_str = "Yes" if row["beats"] else "No"
        margin_str = _fmt(float(row["margin"]))
        lines.append(
            f"| {row['name']} | {_fmt(float(row['baseline_sharpe']))} | "
            f"{_fmt(float(row['model_sharpe']))} | {beats_str} | {margin_str} |"
        )
    beats_all_str = "YES — model beats all baselines" if tearsheet.beats_all_baselines else "NO — model does NOT beat all baselines"
    lines += ["", f"**{beats_all_str}**", ""]

    lines += [
        "## 8. Feature Importance",
        "",
        "| Feature | Mean Importance | Std | Positive-Fold Fraction | Stable? |",
        "|---------|----------------|-----|----------------------|---------|",
    ]
    for feat in importance_result.feature_names:
        mi = importance_result.mean_importance.get(feat, float("nan"))
        si = importance_result.std_importance.get(feat, float("nan"))
        pf = importance_result.positive_fold_fraction.get(feat, float("nan"))
        st = importance_result.is_stable.get(feat, False)
        lines.append(
            f"| {feat} | {_fmt(mi)} | {_fmt(si)} | {_fmt(pf, '.1%')} | {'Yes' if st else 'No'} |"
        )

    lines += [
        "",
        f"Stable features (≥70% positive folds): {', '.join(tearsheet.stable_features) or 'none'}",
        f"Unstable features: {', '.join(tearsheet.unstable_features) or 'none'}",
        "",
        "## 9. Skeptic's Kit Pass/Fail",
        "",
        "| Test | Result | Detail |",
        "|------|--------|--------|",
        f"| Random baseline | {_tick(tearsheet.random_baseline_passed)} | "
        f"margin={_fmt(skeptics_report.random_baseline.margin)}, "
        f"p={skeptics_report.random_baseline.p_value:.2f} |",
        f"| Shuffle test | {_tick(tearsheet.shuffle_test_passed)} | "
        f"margin={_fmt(skeptics_report.shuffle_test.margin)}, "
        f"p={skeptics_report.shuffle_test.p_value:.2f} |",
        f"| Period robustness | {_tick(tearsheet.period_robustness_stable)} | "
        f"ARI mean={_fmt(skeptics_report.period_robustness.ari_mean)}, "
        f"windows={skeptics_report.period_robustness.n_windows} |",
        "",
        "---",
        "",
        "> **STOP:** Review this tearsheet before proceeding to Phase 38 (deployment infrastructure).",
    ]

    out_path.write_text("\n".join(lines))
    logger.info("Honest tearsheet written to %s", out_path)
    return out_path
