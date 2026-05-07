"""Phase 37 edge validation script.

Runs the full Phase 37.1-37.5 validation pipeline on a single asset and
writes all output artefacts to ``results/{asset}/``.

Usage
-----
    uv run python scripts/run_phase37_validation.py --config configs/btc.yaml
    uv run python scripts/run_phase37_validation.py --config configs/btc.yaml \\
        --n-states 8 --train-bars 4000 --test-bars 500 --n-restarts 3

Outputs
-------
    results/{asset}/purged_cv_{date}.parquet
    results/{asset}/combinatorial_cv_{date}.parquet
    results/{asset}/skeptics_report.md
    results/{asset}/honest_tearsheet.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path when run as a script outside the installed package
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.config.loader import load_config
from rde.data.cache import read_cache as read_parquet, write_cache as write_parquet
from rde.data.yfinance_source import YFinanceSource
from rde.evaluation.baselines import run_all_baselines
from rde.evaluation.feature_importance import run_permutation_importance
from rde.evaluation.honest_tearsheet import compute_honest_tearsheet, write_honest_tearsheet
from rde.evaluation.purged_cv import run_combinatorial_purged_cv, run_purged_cv
from rde.evaluation.skeptics import run_full_skeptics_kit, write_skeptics_report
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.inference.viterbi import viterbi_decode
from rde.models.hmm import FittedModel, train_hmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TRANSFORMER_MAP = {
    "LogReturns": lambda params: LogReturns(),
    "RollingVolatility": lambda params: RollingVolatility(window=int(params.get("window", 24))),
    "SmoothedReturns": lambda params: SmoothedReturns(window=int(params.get("window", 12))),
}


def _load_features(
    config_path: Path,
    n_states: int,
    n_restarts: int,
) -> tuple[pd.DataFrame, list[str], str, Path]:
    """Load config, fetch data, build features, return (df, feature_cols, asset, results_dir).

    The feature pipeline is built directly from the config so the validation
    script works correctly for both hourly and daily configs without
    hardcoded column names.
    """
    cfg = load_config(config_path)
    symbol = cfg.asset.symbol

    results_dir = Path(cfg.run.output_dir.format(symbol=symbol))
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = _repo_root / "results" / "cache"
    source = YFinanceSource(cache_dir=cache_dir)
    raw = source.load(symbol, cfg.asset.period, cfg.asset.interval)

    transformers = []
    for fc in cfg.features:
        builder = _TRANSFORMER_MAP.get(fc.name)
        if builder is None:
            raise ValueError(
                f"Unknown feature transformer {fc.name!r}. "
                f"Add it to _TRANSFORMER_MAP in {__file__}."
            )
        transformers.append(builder(fc.params))

    pipeline = FeaturePipeline(transformers)
    df = pipeline.transform(raw)

    # Derive column names from pipeline — works for any window sizes.
    feature_cols = pipeline.output_columns
    return df, feature_cols, symbol, results_dir


def _fit_full_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_states: int,
    n_restarts: int,
) -> tuple[FittedModel, np.ndarray]:
    """Fit a model on the full dataset; return (fitted, viterbi_regimes)."""
    X = df[feature_cols].values
    fitted = train_hmm(
        X, n_states,
        n_restarts=n_restarts,
        feature_names=feature_cols,
        seed_base=42,
    )
    X_scaled = fitted.scaler.transform(X)
    regimes = viterbi_decode(fitted.hmm, X_scaled)
    return fitted, regimes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 37 edge validation")
    parser.add_argument("--config", required=True, type=Path, help="Path to asset YAML config")
    parser.add_argument("--n-states", type=int, default=8, help="Number of HMM states (default 8)")
    parser.add_argument("--train-bars", type=int, default=4000, help="Training window bars (default 4000)")
    parser.add_argument("--test-bars", type=int, default=500, help="Test window bars (default 500)")
    parser.add_argument("--embargo-bars", type=int, default=24, help="Embargo gap bars (default 24)")
    parser.add_argument("--n-restarts", type=int, default=3, help="HMM restart count (default 3)")
    parser.add_argument("--n-sims", type=int, default=100, help="Null simulation count (default 100)")
    parser.add_argument("--n-permutations", type=int, default=10, help="Permutation count per fold feature (default 10)")
    parser.add_argument(
        "--ann-factor", type=int, default=8760,
        help="Bars per year for annualisation (default 8760 for hourly; use 252 for daily)",
    )
    parser.add_argument(
        "--daily-volume-usd", type=float, default=20_000_000_000.0,
        help="Asset daily trading volume in USD for capacity estimate (default 20B for BTC)",
    )
    parser.add_argument(
        "--period-robustness-window-bars", type=int, default=4383,
        help=(
            "Period robustness window length in bars "
            "(default 4383 ≈ 6mo hourly; use ~504 for 24mo daily). "
            "Auto-scaled if larger than dataset."
        ),
    )
    parser.add_argument(
        "--period-robustness-step-bars", type=int, default=730,
        help=(
            "Period robustness step between windows "
            "(default 730 ≈ 1mo hourly; use ~126 for 6mo daily)."
        ),
    )
    args = parser.parse_args()

    logger.info("=== Phase 37 Validation starting ===")
    logger.info("Config: %s", args.config)
    logger.info("n_states=%d train_bars=%d test_bars=%d embargo=%d n_restarts=%d ann_factor=%d",
                args.n_states, args.train_bars, args.test_bars, args.embargo_bars, args.n_restarts,
                args.ann_factor)

    # --- 1. Load features ---
    df, feature_cols, symbol, results_dir = _load_features(args.config, args.n_states, args.n_restarts)
    logger.info("Loaded %d bars for %s. Features: %s", len(df), symbol, feature_cols)

    train_kwargs = {"n_restarts": args.n_restarts, "n_iter": 200, "seed_base": 42}

    # --- 2. Fit full model for skeptics + full-dataset Viterbi ---
    logger.info("Fitting full-dataset model (%d states, %d restarts)…", args.n_states, args.n_restarts)
    fitted_full, regimes_full = _fit_full_model(df, feature_cols, args.n_states, args.n_restarts)
    logger.info("Full model BIC=%.1f", fitted_full.bic)

    from rde.analysis.backtest import backtest_tearsheet, run_backtest
    from rde.evaluation.purged_cv import _default_strategy_config
    strategy_full = _default_strategy_config(fitted_full, args.ann_factor, 0.0001)
    bt_full = run_backtest(df["log_return"], regimes_full, strategy_full)
    ts_full = backtest_tearsheet(bt_full, ann_factor=args.ann_factor)
    model_sharpe_full = float(ts_full["sharpe"])
    logger.info("Full-dataset model Sharpe=%.3f", model_sharpe_full)

    # --- 3. Purged k-fold CV ---
    logger.info("Running purged k-fold CV…")
    fold_results, cv_summary = run_purged_cv(
        df, feature_cols, "log_return", args.n_states,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        embargo_bars=args.embargo_bars,
        ann_factor=args.ann_factor,
        transaction_cost=0.0001,
        train_kwargs=train_kwargs,
        results_dir=results_dir,
        asset=symbol,
    )
    cv_sharpes = [fr.sharpe for fr in fold_results if np.isfinite(fr.sharpe)]
    logger.info(
        "Purged CV: %d folds, Sharpe mean=%.3f ± %.3f",
        len(fold_results),
        float(np.mean(cv_sharpes)) if cv_sharpes else float("nan"),
        float(np.std(cv_sharpes)) if len(cv_sharpes) > 1 else 0.0,
    )

    # --- 4. Combinatorial purged CV ---
    logger.info("Running combinatorial purged CV…")
    combo_results, combo_summary = run_combinatorial_purged_cv(
        df, feature_cols, "log_return", args.n_states,
        n_groups=6, n_test_groups=2,
        embargo_bars=args.embargo_bars,
        ann_factor=args.ann_factor,
        transaction_cost=0.0001,
        train_kwargs=train_kwargs,
        results_dir=results_dir,
        asset=symbol,
    )
    logger.info("Combinatorial CV: %d folds", len(combo_results))

    # --- 5. Skeptic's kit ---
    logger.info("Running skeptic's kit…")
    skeptics = run_full_skeptics_kit(
        df, feature_cols, "log_return", args.n_states,
        model_sharpe=model_sharpe_full,
        regimes=regimes_full,
        ann_factor=args.ann_factor,
        n_sims=args.n_sims,
        transaction_cost=0.0001,
        train_kwargs=train_kwargs,
        seed=42,
        period_robustness_window_bars=args.period_robustness_window_bars,
        period_robustness_step_bars=args.period_robustness_step_bars,
    )
    skeptics_path = write_skeptics_report(skeptics, results_dir, symbol)
    logger.info("Skeptics report → %s", skeptics_path)

    # --- 6. Baselines ---
    logger.info("Running baselines…")
    baselines = run_all_baselines(
        df, feature_cols, "log_return",
        ann_factor=args.ann_factor,
        transaction_cost=0.0001,
        train_kwargs=train_kwargs,
    )
    for name, br in baselines.items():
        logger.info("Baseline %s: Sharpe=%.3f", name, br.sharpe)

    # --- 7. Permutation feature importance ---
    logger.info("Running permutation feature importance…")
    importance = run_permutation_importance(
        df, feature_cols, "log_return", args.n_states,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        embargo_bars=args.embargo_bars,
        n_permutations=args.n_permutations,
        ann_factor=args.ann_factor,
        transaction_cost=0.0001,
        train_kwargs=train_kwargs,
        seed=42,
    )
    for feat in feature_cols:
        logger.info(
            "Importance %s: mean=%.3f, pos_frac=%.2f, stable=%s",
            feat,
            importance.mean_importance.get(feat, float("nan")),
            importance.positive_fold_fraction.get(feat, float("nan")),
            importance.is_stable.get(feat, False),
        )

    # --- 8. Honest tearsheet ---
    logger.info("Computing honest tearsheet…")
    tearsheet = compute_honest_tearsheet(
        fold_results, baselines, skeptics, importance,
        daily_volume_usd=args.daily_volume_usd,
        asset=symbol,
        ann_factor=args.ann_factor,
    )
    tearsheet_path = write_honest_tearsheet(
        tearsheet, fold_results, baselines, skeptics, importance, results_dir,
        ann_factor=args.ann_factor,
    )
    logger.info("Honest tearsheet → %s", tearsheet_path)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"PHASE 37 VALIDATION COMPLETE — {symbol}")
    print("=" * 60)
    print(f"CV Folds:        {tearsheet.n_cv_folds}")
    print(f"Sharpe mean±std: {tearsheet.sharpe_mean:.3f} ± {tearsheet.sharpe_std:.3f}")
    print(f"Sharpe worst 5%: {tearsheet.sharpe_worst_5pct:.3f}")
    print(f"MDD worst 5%:    {tearsheet.mdd_worst_5pct:.1%}")
    print(f"Cost break-even: {tearsheet.cost_break_even_bps:.1f} bps")
    print(f"Random baseline: {'PASS' if tearsheet.random_baseline_passed else 'FAIL'}")
    print(f"Shuffle test:    {'PASS' if tearsheet.shuffle_test_passed else 'FAIL'}")
    print(f"Period robust:   {'PASS' if tearsheet.period_robustness_stable else 'FAIL'}")
    print(f"Beats baselines: {'ALL' if tearsheet.beats_all_baselines else 'PARTIAL'}")
    print(f"Stable features: {', '.join(tearsheet.stable_features) or 'none'}")
    print(f"Capacity est.:   ${tearsheet.capacity_estimate_usd:,.0f}")
    print()
    print(f"Tearsheet:       {tearsheet_path}")
    print(f"Skeptics report: {skeptics_path}")
    print()
    print("STOP — review honest_tearsheet.md before proceeding to Phase 38.")
    print("=" * 60)


if __name__ == "__main__":
    main()
