"""Skeptic's kit for adversarial edge validation (Phase 37.2).

Provides a battery of tests that try to falsify the model's edge:

1. **Random baseline** — replace HMM labels with random regimes having the
   same marginal distribution. If the model can't beat noise, it has no edge.
2. **Shuffle test** — permute regime labels, destroying temporal structure.
3. **Feature ablation** — refit the model dropping one feature at a time.
4. **Period robustness** — refit on rolling windows; measure ARI and parameter
   dispersion.
5. **Cost sensitivity** — sweep transaction cost; find break-even.
6. **Slippage stress** — apply execution slippage; find break-even.

All results are collected into a :class:`SkepticsReport` and optionally
written to ``results/{asset}/skeptics_report.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rde.analysis.backtest import (
    BacktestResult,
    RegimeRule,
    RegimeStrategyConfig,
    backtest_tearsheet,
    run_backtest,
)
from rde.evaluation.purged_cv import (
    _align_state_order,
    _compute_parameter_dispersion,
    _default_strategy_config,
)
from rde.inference.online import OnlineDecoder
from rde.models.hmm import FittedModel, train_hmm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NullTestResult:
    """Result of a null-hypothesis test (random baseline or shuffle test).

    Attributes
    ----------
    name : str
        Test identifier — ``"random_baseline"`` or ``"shuffle_test"``.
    model_sharpe : float
        Sharpe of the model strategy being tested against.
    null_sharpe_mean : float
        Mean Sharpe across null simulation runs.
    null_sharpe_std : float
        Std of null Sharpe.
    p_value : float
        Fraction of null runs whose Sharpe >= ``model_sharpe``.
    margin : float
        ``model_sharpe - null_sharpe_mean``.
    passed : bool
        ``True`` if ``margin >= 0.3`` (model is meaningfully better than noise).
    """

    name: str
    model_sharpe: float
    null_sharpe_mean: float
    null_sharpe_std: float
    p_value: float
    margin: float
    passed: bool


@dataclass
class AblationResult:
    """Result of dropping one feature and refitting the model.

    Attributes
    ----------
    feature : str
        The dropped feature name.
    sharpe_without : float
        Sharpe when the model is retrained without this feature.
    importance : float
        ``model_sharpe - sharpe_without``. Positive = feature adds value.
    """

    feature: str
    sharpe_without: float
    importance: float


@dataclass
class PeriodRobustnessResult:
    """Stability of regime structure across rolling time windows.

    Attributes
    ----------
    n_windows : int
        Number of rolling windows fitted.
    ari_mean : float
        Mean pairwise ARI across all window pairs.
    transmat_frobenius_std : float
        Std of aligned transition-matrix Frobenius distances across windows.
    means_frobenius_std : float
        Std of aligned emission-means Frobenius distances.
    is_stable : bool
        ``True`` if ``ari_mean >= 0.7`` and ``transmat_frobenius_std < 0.1``.
    """

    n_windows: int
    ari_mean: float
    transmat_frobenius_std: float
    means_frobenius_std: float
    is_stable: bool


@dataclass
class CostSweepResult:
    """Sharpe as a function of transaction cost or slippage.

    Attributes
    ----------
    cost_bps : list[float]
        The cost levels tested (basis points per side).
    sharpes : list[float]
        Sharpe at each cost level.
    break_even_bps : float
        Interpolated cost level where Sharpe = 0. ``inf`` if Sharpe is always
        positive within the sweep range.
    """

    cost_bps: list[float]
    sharpes: list[float]
    break_even_bps: float


@dataclass
class SkepticsReport:
    """Aggregated results from the full skeptic's kit.

    Attributes
    ----------
    random_baseline : NullTestResult
    shuffle_test : NullTestResult
    ablation : list[AblationResult]
        One entry per feature.
    period_robustness : PeriodRobustnessResult
    cost_sensitivity : CostSweepResult
    slippage_stress : CostSweepResult
    overall_passed : bool
        ``True`` if all critical tests pass (random baseline + shuffle test +
        period robustness).
    """

    random_baseline: NullTestResult
    shuffle_test: NullTestResult
    ablation: list[AblationResult]
    period_robustness: PeriodRobustnessResult
    cost_sensitivity: CostSweepResult
    slippage_stress: CostSweepResult
    overall_passed: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sharpe_from_backtest(bt: BacktestResult, ann_factor: int) -> float:
    ts = backtest_tearsheet(bt, ann_factor=ann_factor)
    return float(ts["sharpe"])


def _run_null_sims(
    returns: np.ndarray,
    null_regimes_fn,
    strategy_config: RegimeStrategyConfig,
    ann_factor: int,
    n_sims: int,
    rng: np.random.Generator,
) -> list[float]:
    """Run n_sims null simulations and return their Sharpe values."""
    sharpes: list[float] = []
    ret_series = pd.Series(returns)
    for _ in range(n_sims):
        null_regimes = null_regimes_fn(rng)
        bt = run_backtest(ret_series, null_regimes, strategy_config)
        sharpes.append(_sharpe_from_backtest(bt, ann_factor))
    return sharpes


def _interpolate_break_even(cost_bps: list[float], sharpes: list[float]) -> float:
    """Find cost level where Sharpe crosses zero by linear interpolation."""
    for i in range(len(sharpes) - 1):
        if sharpes[i] >= 0 and sharpes[i + 1] < 0:
            # Linear interpolation between i and i+1
            slope = (sharpes[i + 1] - sharpes[i]) / (cost_bps[i + 1] - cost_bps[i])
            return float(cost_bps[i] - sharpes[i] / slope)
    if all(s >= 0 for s in sharpes):
        return float("inf")
    return float(cost_bps[0])  # Already negative at lowest cost


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def run_random_baseline(
    returns: np.ndarray,
    regimes: np.ndarray,
    strategy_config: RegimeStrategyConfig,
    ann_factor: int = 8760,
    n_sims: int = 100,
    seed: int = 42,
) -> NullTestResult:
    """Test whether a random regime sequence matches the model's Sharpe.

    Generates ``n_sims`` random regime sequences drawn from the empirical
    marginal distribution (same state counts on average, no temporal
    structure). Runs the same strategy rules. Returns a :class:`NullTestResult`
    with ``p_value = fraction of null Sharpes >= model_sharpe``.

    Parameters
    ----------
    returns : np.ndarray
        Per-bar asset returns, shape (T,).
    regimes : np.ndarray
        Model regime labels, shape (T,), used for computing model Sharpe and
        for the empirical marginal distribution.
    strategy_config : RegimeStrategyConfig
        Strategy rules to apply in every simulation.
    ann_factor : int
        Bars per year for annualisation.
    n_sims : int
        Number of null simulations.
    seed : int
        RNG seed.

    Returns
    -------
    NullTestResult
    """
    rng = np.random.default_rng(seed)
    ret_series = pd.Series(returns)
    model_bt = run_backtest(ret_series, regimes, strategy_config)
    model_sharpe = _sharpe_from_backtest(model_bt, ann_factor)

    n_states = int(regimes.max()) + 1
    counts = np.bincount(regimes, minlength=n_states)
    probs = counts / counts.sum()

    def null_fn(r: np.random.Generator) -> np.ndarray:
        return r.choice(n_states, size=len(regimes), p=probs)

    null_sharpes = _run_null_sims(returns, null_fn, strategy_config, ann_factor, n_sims, rng)

    null_mean = float(np.mean(null_sharpes))
    null_std = float(np.std(null_sharpes, ddof=1)) if len(null_sharpes) > 1 else 0.0
    p_value = float(np.mean([s >= model_sharpe for s in null_sharpes]))
    margin = model_sharpe - null_mean

    logger.info(
        "Random baseline: model_sharpe=%.3f null_mean=%.3f margin=%.3f p_value=%.3f",
        model_sharpe, null_mean, margin, p_value,
    )
    return NullTestResult(
        name="random_baseline",
        model_sharpe=model_sharpe,
        null_sharpe_mean=null_mean,
        null_sharpe_std=null_std,
        p_value=p_value,
        margin=margin,
        passed=margin >= 0.3,
    )


def run_shuffle_test(
    returns: np.ndarray,
    regimes: np.ndarray,
    strategy_config: RegimeStrategyConfig,
    ann_factor: int = 8760,
    n_sims: int = 100,
    seed: int = 42,
) -> NullTestResult:
    """Test whether permuted regime labels match the model's Sharpe.

    Permutes the regime label sequence (preserving marginal counts but
    destroying temporal structure) ``n_sims`` times and computes the
    Sharpe of the same strategy rules. If the model's temporal structure
    does not matter, the model has no edge.

    Parameters
    ----------
    returns : np.ndarray
    regimes : np.ndarray
    strategy_config : RegimeStrategyConfig
    ann_factor : int
    n_sims : int
    seed : int

    Returns
    -------
    NullTestResult
    """
    rng = np.random.default_rng(seed)
    ret_series = pd.Series(returns)
    model_bt = run_backtest(ret_series, regimes, strategy_config)
    model_sharpe = _sharpe_from_backtest(model_bt, ann_factor)

    def null_fn(r: np.random.Generator) -> np.ndarray:
        shuffled = regimes.copy()
        r.shuffle(shuffled)
        return shuffled

    null_sharpes = _run_null_sims(returns, null_fn, strategy_config, ann_factor, n_sims, rng)

    null_mean = float(np.mean(null_sharpes))
    null_std = float(np.std(null_sharpes, ddof=1)) if len(null_sharpes) > 1 else 0.0
    p_value = float(np.mean([s >= model_sharpe for s in null_sharpes]))
    margin = model_sharpe - null_mean

    logger.info(
        "Shuffle test: model_sharpe=%.3f null_mean=%.3f margin=%.3f p_value=%.3f",
        model_sharpe, null_mean, margin, p_value,
    )
    return NullTestResult(
        name="shuffle_test",
        model_sharpe=model_sharpe,
        null_sharpe_mean=null_mean,
        null_sharpe_std=null_std,
        p_value=p_value,
        margin=margin,
        passed=margin >= 0.3,
    )


def run_feature_ablation(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    model_sharpe: float,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_frac: float = 0.7,
    train_kwargs: dict | None = None,
) -> list[AblationResult]:
    """Refit the model dropping one feature at a time; measure Sharpe degradation.

    Uses a simple (non-purged) ``train_frac`` / ``1-train_frac`` split for
    speed. For each feature subset (drop one), fits the model, decodes the
    test window causally via :class:`OnlineDecoder`, and backtests with the
    same default strategy.

    Parameters
    ----------
    df_features : pd.DataFrame
    feature_cols : list[str]
    returns_col : str
    n_states : int
    model_sharpe : float
        Sharpe of the full-feature model (baseline for importance scores).
    ann_factor : int
    transaction_cost : float
    train_frac : float
    train_kwargs : dict | None

    Returns
    -------
    list[AblationResult]
        One entry per feature in ``feature_cols``.
    """
    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    n = len(df_features)
    split = int(n * train_frac)
    ret_test = df_features[returns_col].values[split:]
    ret_test_series = pd.Series(ret_test)

    results: list[AblationResult] = []
    for drop_feat in feature_cols:
        reduced = [c for c in feature_cols if c != drop_feat]
        if not reduced:
            continue
        kw["feature_names"] = reduced
        X_all = df_features[reduced].values
        X_train = X_all[:split]
        X_test = X_all[split:]

        try:
            fitted = train_hmm(X_train, n_states, **kw)
        except Exception:
            logger.warning("Ablation: train_hmm failed dropping %s", drop_feat)
            results.append(AblationResult(feature=drop_feat, sharpe_without=float("nan"), importance=float("nan")))
            continue

        strategy = _default_strategy_config(fitted, ann_factor, transaction_cost)
        decoder = OnlineDecoder(fitted)
        posteriors = decoder.batch_filter(X_test)
        regimes = posteriors.argmax(axis=1)

        bt = run_backtest(ret_test_series, regimes, strategy)
        sharpe_without = _sharpe_from_backtest(bt, ann_factor)
        importance = model_sharpe - sharpe_without

        logger.info("Ablation drop=%s sharpe_without=%.3f importance=%.3f", drop_feat, sharpe_without, importance)
        results.append(AblationResult(feature=drop_feat, sharpe_without=sharpe_without, importance=importance))

    return results


def run_period_robustness(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    n_states: int,
    window_bars: int = 4383,
    step_bars: int = 730,
    train_kwargs: dict | None = None,
) -> PeriodRobustnessResult:
    """Assess stability of HMM parameters across rolling time windows.

    Refits the HMM on each rolling ``window_bars`` slice, advancing by
    ``step_bars``. Computes pairwise ARI between decoded state sequences
    (measuring label stability) and Frobenius dispersion of aligned transition
    matrices (measuring parameter stability).

    Parameters
    ----------
    df_features : pd.DataFrame
    feature_cols : list[str]
    n_states : int
    window_bars : int
        Window length (default ~6 months of hourly data).
    step_bars : int
        Step between windows (default ~1 month).
    train_kwargs : dict | None

    Returns
    -------
    PeriodRobustnessResult
    """
    from sklearn.metrics import adjusted_rand_score
    from rde.inference.viterbi import viterbi_decode

    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    X_all = df_features[feature_cols].values
    n = len(X_all)

    transmats: list[np.ndarray] = []
    means_list: list[np.ndarray] = []
    sequences: list[np.ndarray] = []

    start = 0
    while start + window_bars <= n:
        X_win = X_all[start: start + window_bars]
        try:
            fitted = train_hmm(X_win, n_states, **kw)
        except Exception:
            logger.warning("Period robustness: train_hmm failed at start=%d", start)
            start += step_bars
            continue

        X_scaled = fitted.scaler.transform(X_win)
        states = viterbi_decode(fitted.hmm, X_scaled)
        transmats.append(fitted.hmm.transmat_.copy())
        means_list.append(fitted.hmm.means_.copy())
        sequences.append(states)
        start += step_bars

    n_windows = len(sequences)
    if n_windows < 2:
        logger.warning("Period robustness: only %d windows fitted, need >= 2.", n_windows)
        return PeriodRobustnessResult(
            n_windows=n_windows,
            ari_mean=float("nan"),
            transmat_frobenius_std=float("nan"),
            means_frobenius_std=float("nan"),
            is_stable=False,
        )

    # Pairwise ARI — only valid for same-length sequences, use min length
    ari_values: list[float] = []
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            min_len = min(len(sequences[i]), len(sequences[j]))
            ari_values.append(float(adjusted_rand_score(sequences[i][:min_len], sequences[j][:min_len])))

    ari_mean = float(np.mean(ari_values))

    ret_idx = feature_cols.index("log_return") if "log_return" in feature_cols else 0
    t_dists, m_dists = _compute_parameter_dispersion(transmats, means_list, ret_idx)
    transmat_std = float(np.std(t_dists, ddof=1)) if len(t_dists) > 1 else 0.0
    means_std = float(np.std(m_dists, ddof=1)) if len(m_dists) > 1 else 0.0

    is_stable = ari_mean >= 0.7 and transmat_std < 0.1
    logger.info(
        "Period robustness: n_windows=%d ari_mean=%.3f transmat_std=%.3f is_stable=%s",
        n_windows, ari_mean, transmat_std, is_stable,
    )
    return PeriodRobustnessResult(
        n_windows=n_windows,
        ari_mean=ari_mean,
        transmat_frobenius_std=transmat_std,
        means_frobenius_std=means_std,
        is_stable=is_stable,
    )


def run_cost_sensitivity(
    returns: np.ndarray,
    regimes: np.ndarray,
    base_config: RegimeStrategyConfig,
    ann_factor: int = 8760,
    cost_bps_range: list[float] | None = None,
) -> CostSweepResult:
    """Sweep transaction cost and measure Sharpe degradation.

    Parameters
    ----------
    returns : np.ndarray
    regimes : np.ndarray
    base_config : RegimeStrategyConfig
        Base strategy (rules are reused; ``transaction_cost`` is overridden).
    ann_factor : int
    cost_bps_range : list[float] | None
        Costs to test in basis points (default [1, 5, 10, 15, 20, 25]).

    Returns
    -------
    CostSweepResult
    """
    if cost_bps_range is None:
        cost_bps_range = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    ret_series = pd.Series(returns)
    sharpes: list[float] = []
    for bps in cost_bps_range:
        cfg = RegimeStrategyConfig(
            rules=list(base_config.rules),
            transaction_cost=bps / 10_000.0,
            ann_factor=ann_factor,
        )
        bt = run_backtest(ret_series, regimes, cfg)
        sharpes.append(_sharpe_from_backtest(bt, ann_factor))
        logger.debug("Cost sweep: %.1f bps → sharpe=%.3f", bps, sharpes[-1])

    break_even = _interpolate_break_even(cost_bps_range, sharpes)
    logger.info("Cost sensitivity: break_even=%.1f bps", break_even)
    return CostSweepResult(cost_bps=list(cost_bps_range), sharpes=sharpes, break_even_bps=break_even)


def run_slippage_stress(
    returns: np.ndarray,
    positions: np.ndarray,
    regimes: np.ndarray,
    base_config: RegimeStrategyConfig,
    ann_factor: int = 8760,
    slippage_bps_range: list[float] | None = None,
) -> CostSweepResult:
    """Apply execution slippage and measure Sharpe degradation.

    At each bar, the adjusted return is:
    ``adjusted_return[t] = return[t] - |delta_position[t]| * slippage_bps / 10000``

    where ``delta_position`` is computed from the provided ``positions`` array.

    Parameters
    ----------
    returns : np.ndarray
    positions : np.ndarray
        Position at each bar (from a prior backtest).
    regimes : np.ndarray
    base_config : RegimeStrategyConfig
    ann_factor : int
    slippage_bps_range : list[float] | None
        Slippage levels in bps (default [0, 10, 20, 30, 40, 50]).

    Returns
    -------
    CostSweepResult
    """
    if slippage_bps_range is None:
        slippage_bps_range = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    delta_pos = np.abs(np.diff(np.concatenate([[0.0], positions])))
    sharpes: list[float] = []

    for bps in slippage_bps_range:
        slippage_cost = delta_pos * bps / 10_000.0
        adj_returns = returns - slippage_cost
        cfg = RegimeStrategyConfig(
            rules=list(base_config.rules),
            transaction_cost=0.0,  # slippage already baked in
            ann_factor=ann_factor,
        )
        bt = run_backtest(pd.Series(adj_returns), regimes, cfg)
        sharpes.append(_sharpe_from_backtest(bt, ann_factor))
        logger.debug("Slippage sweep: %.1f bps → sharpe=%.3f", bps, sharpes[-1])

    break_even = _interpolate_break_even(slippage_bps_range, sharpes)
    logger.info("Slippage stress: break_even=%.1f bps", break_even)
    return CostSweepResult(cost_bps=list(slippage_bps_range), sharpes=sharpes, break_even_bps=break_even)


def run_full_skeptics_kit(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    model_sharpe: float,
    regimes: np.ndarray,
    ann_factor: int = 8760,
    n_sims: int = 100,
    transaction_cost: float = 0.0001,
    train_kwargs: dict | None = None,
    seed: int = 42,
) -> SkepticsReport:
    """Orchestrate all skeptic's tests and return a :class:`SkepticsReport`.

    Parameters
    ----------
    df_features : pd.DataFrame
        Full feature + return DataFrame with DatetimeIndex.
    feature_cols : list[str]
    returns_col : str
    n_states : int
    model_sharpe : float
        Sharpe of the model strategy on the full dataset.
    regimes : np.ndarray
        Full in-sample Viterbi / causal-decoded regime sequence, shape (T,).
    ann_factor : int
    n_sims : int
        Null simulation count for random baseline and shuffle test.
    transaction_cost : float
    train_kwargs : dict | None
    seed : int

    Returns
    -------
    SkepticsReport
    """
    returns = df_features[returns_col].values
    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    # Build a reference strategy config using a simple full model fit on first 70%
    n = len(df_features)
    split = int(n * 0.7)
    X_train = df_features[feature_cols].values[:split]
    fitted_ref = train_hmm(X_train, n_states, **kw)
    strategy_config = _default_strategy_config(fitted_ref, ann_factor, transaction_cost)

    # Run backtest with model regimes to get positions for slippage test
    ret_series = pd.Series(returns)
    model_bt = run_backtest(ret_series, regimes, strategy_config)
    positions = model_bt.positions.values

    logger.info("--- Skeptic's Kit starting ---")

    random_result = run_random_baseline(
        returns, regimes, strategy_config, ann_factor=ann_factor, n_sims=n_sims, seed=seed,
    )
    shuffle_result = run_shuffle_test(
        returns, regimes, strategy_config, ann_factor=ann_factor, n_sims=n_sims, seed=seed + 1,
    )
    ablation = run_feature_ablation(
        df_features, feature_cols, returns_col, n_states, model_sharpe,
        ann_factor=ann_factor, transaction_cost=transaction_cost,
        train_kwargs=train_kwargs,
    )
    period_robustness = run_period_robustness(
        df_features, feature_cols, n_states, train_kwargs=train_kwargs,
    )
    cost_sensitivity = run_cost_sensitivity(
        returns, regimes, strategy_config, ann_factor=ann_factor,
    )
    slippage_stress = run_slippage_stress(
        returns, positions, regimes, strategy_config, ann_factor=ann_factor,
    )

    overall_passed = (
        random_result.passed
        and shuffle_result.passed
        and period_robustness.is_stable
    )
    logger.info("--- Skeptic's Kit complete: overall_passed=%s ---", overall_passed)

    return SkepticsReport(
        random_baseline=random_result,
        shuffle_test=shuffle_result,
        ablation=ablation,
        period_robustness=period_robustness,
        cost_sensitivity=cost_sensitivity,
        slippage_stress=slippage_stress,
        overall_passed=overall_passed,
    )


def write_skeptics_report(
    report: SkepticsReport,
    results_dir: Path,
    asset: str,
) -> Path:
    """Write the skeptic's kit results to a Markdown file.

    Parameters
    ----------
    report : SkepticsReport
    results_dir : Path
        Directory to write into (e.g. ``results/BTC-USD/``).
    asset : str
        Asset label for the report header.

    Returns
    -------
    Path
        Path to the written file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "skeptics_report.md"

    def _tick(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    lines: list[str] = [
        f"# Skeptic's Kit Report — {asset}",
        "",
        "## Summary",
        "",
        f"| Test | Result |",
        f"|------|--------|",
        f"| Random baseline | {_tick(report.random_baseline.passed)} |",
        f"| Shuffle test | {_tick(report.shuffle_test.passed)} |",
        f"| Period robustness | {_tick(report.period_robustness.is_stable)} |",
        f"| **Overall** | **{_tick(report.overall_passed)}** |",
        "",
        "## 1. Random Baseline",
        "",
        f"- Model Sharpe: {report.random_baseline.model_sharpe:.3f}",
        f"- Null mean Sharpe: {report.random_baseline.null_sharpe_mean:.3f} "
        f"± {report.random_baseline.null_sharpe_std:.3f}",
        f"- Margin (model − null mean): {report.random_baseline.margin:.3f}",
        f"- p-value (fraction null ≥ model): {report.random_baseline.p_value:.3f}",
        f"- Result: **{_tick(report.random_baseline.passed)}** (threshold: margin ≥ 0.3)",
        "",
        "## 2. Shuffle Test",
        "",
        f"- Model Sharpe: {report.shuffle_test.model_sharpe:.3f}",
        f"- Null mean Sharpe: {report.shuffle_test.null_sharpe_mean:.3f} "
        f"± {report.shuffle_test.null_sharpe_std:.3f}",
        f"- Margin: {report.shuffle_test.margin:.3f}",
        f"- p-value: {report.shuffle_test.p_value:.3f}",
        f"- Result: **{_tick(report.shuffle_test.passed)}** (threshold: margin ≥ 0.3)",
        "",
        "## 3. Feature Ablation",
        "",
        "| Feature | Sharpe without | Importance |",
        "|---------|---------------|------------|",
    ]
    for ab in report.ablation:
        imp_str = f"{ab.importance:+.3f}" if np.isfinite(ab.importance) else "N/A"
        sw_str = f"{ab.sharpe_without:.3f}" if np.isfinite(ab.sharpe_without) else "N/A"
        lines.append(f"| {ab.feature} | {sw_str} | {imp_str} |")

    lines += [
        "",
        "## 4. Period Robustness",
        "",
        f"- Windows fitted: {report.period_robustness.n_windows}",
        f"- Mean pairwise ARI: {report.period_robustness.ari_mean:.3f} (threshold ≥ 0.7)",
        f"- Transition-matrix Frobenius std: "
        f"{report.period_robustness.transmat_frobenius_std:.3f} (threshold < 0.1)",
        f"- Emission-means Frobenius std: {report.period_robustness.means_frobenius_std:.3f}",
        f"- Result: **{_tick(report.period_robustness.is_stable)}**",
        "",
        "## 5. Cost Sensitivity",
        "",
        "| Cost (bps/side) | Sharpe |",
        "|-----------------|--------|",
    ]
    for bps, sh in zip(report.cost_sensitivity.cost_bps, report.cost_sensitivity.sharpes):
        lines.append(f"| {bps:.0f} | {sh:.3f} |")
    be = report.cost_sensitivity.break_even_bps
    be_str = f"{be:.1f}" if np.isfinite(be) else "∞"
    lines += ["", f"**Break-even cost: {be_str} bps/side**", ""]

    lines += [
        "## 6. Slippage Stress",
        "",
        "| Slippage (bps) | Sharpe |",
        "|----------------|--------|",
    ]
    for bps, sh in zip(report.slippage_stress.cost_bps, report.slippage_stress.sharpes):
        lines.append(f"| {bps:.0f} | {sh:.3f} |")
    be2 = report.slippage_stress.break_even_bps
    be2_str = f"{be2:.1f}" if np.isfinite(be2) else "∞"
    lines += ["", f"**Break-even slippage: {be2_str} bps**", ""]

    out_path.write_text("\n".join(lines))
    logger.info("Skeptics report written to %s", out_path)
    return out_path
