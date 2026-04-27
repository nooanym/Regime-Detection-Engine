"""CLI entry point: ``rde run --config <path>`` runs the full detection pipeline."""

import logging
import sys
from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rde.config.loader import load_config
from rde.data.yfinance_source import YFinanceSource
from rde.evaluation.persistence import empirical_dwell_times, expected_dwell_times
from rde.evaluation.regime_analytics import (
    compute_regime_stats,
    regime_stats_to_dataframe,
    regime_transition_table,
)
from rde.evaluation.stability import stability_across_restarts
from rde.evaluation.transition import stationary_distribution, transition_entropy
from rde.evaluation.walk_forward import WalkForwardHarness
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.features.volume import LogVolume, RollingVolumeZScore
from rde.inference.smoothing import forward_backward_posteriors
from rde.inference.viterbi import viterbi_decode
from rde.labeling.ranking import HEURISTIC_DISCLAIMER, LabelledState, rank_states
from rde.models.hmm import FittedModel, train_hmm
from rde.models.selection import select_n_states
from rde.viz.interactive import (
    plot_per_regime_returns as iplot_per_regime_returns,
    plot_price_with_regimes as iplot_price_with_regimes,
    plot_regime_timeline as iplot_regime_timeline,
    plot_transition_heatmap as iplot_transition_heatmap,
)
from rde.viz.static_plots import (
    plot_per_regime_returns,
    plot_price_with_regimes,
    plot_regime_timeline,
    plot_transition_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

_FEATURE_REGISTRY: dict = {
    "LogReturns": lambda params: LogReturns(),
    "RollingVolatility": lambda params: RollingVolatility(**params),
    "SmoothedReturns": lambda params: SmoothedReturns(**params),
    "LogVolume": lambda params: LogVolume(),
    "RollingVolumeZScore": lambda params: RollingVolumeZScore(**params),
}


def _build_pipeline(feature_configs) -> FeaturePipeline:
    transformers = []
    for fc in feature_configs:
        if fc.name not in _FEATURE_REGISTRY:
            raise ValueError(
                f"Unknown feature transformer: {fc.name!r}. "
                f"Available: {list(_FEATURE_REGISTRY)}"
            )
        transformers.append(_FEATURE_REGISTRY[fc.name](fc.params))
    return FeaturePipeline(transformers)


def _write_diagnostics(
    path: Path,
    symbol: str,
    fitted: FittedModel,
    states: np.ndarray,
    df_features: pd.DataFrame,
    labelled_states: list[LabelledState],
    scores_df: pd.DataFrame | None,
    stationary: np.ndarray,
    stability_result: dict | None = None,
    wf_result: pd.DataFrame | None = None,
    wf_cfg: object | None = None,
    regime_stats_df: pd.DataFrame | None = None,
    transition_table: pd.DataFrame | None = None,
) -> None:
    lines: list[str] = []
    sep = "=" * 70

    from rde.models.student_t_hmm import StudentTHMM as _StudentTHMM
    emission_label = "student_t" if isinstance(fitted.hmm, _StudentTHMM) else "gaussian"

    lines += [
        sep,
        "REGIME DETECTION ENGINE — DIAGNOSTICS",
        f"Asset:         {symbol}",
        f"Date range:    {df_features.index[0]}  —  {df_features.index[-1]}",
        f"Observations:  {len(df_features)}",
        f"Features:      {fitted.feature_names}",
        f"n_states:      {fitted.n_states}",
        f"Emission:      {emission_label}",
        f"Seed (winner): {fitted.seed}",
        f"Log-lik:       {fitted.log_likelihood:.4f}",
        f"AIC:           {fitted.aic:.4f}",
        f"BIC:           {fitted.bic:.4f}",
        "",
    ]

    lines.append("TRANSITION MATRIX")
    header = "       " + "  ".join(f"  S{i}" for i in range(fitted.n_states))
    lines.append(header)
    for i, row in enumerate(fitted.hmm.transmat_):
        vals = "  ".join(f"{v:.4f}" for v in row)
        lines.append(f"  S{i}    {vals}")
    lines.append("")

    lines.append("STATIONARY DISTRIBUTION")
    for i, p in enumerate(stationary):
        lines.append(f"  State {i}: {p:.4f}")
    lines.append("")

    lines.append("STATE MEANS  (in scaled feature space)")
    from rde.models.student_t_hmm import StudentTHMM as _StudentTHMMInner
    is_student_t = isinstance(fitted.hmm, _StudentTHMMInner)
    for i in range(fitted.n_states):
        parts = "  ".join(
            f"{name}={val:.4f}"
            for name, val in zip(fitted.feature_names, fitted.hmm.means_[i])
        )
        dof_str = f"  dof={fitted.hmm.dofs_[i]:.2f}" if is_student_t else ""
        lines.append(f"  State {i}: {parts}{dof_str}")
    lines.append("")

    lines.append("TRANSITION ENTROPY (nats per state)")
    ent = transition_entropy(fitted.hmm.transmat_)
    for i, e in enumerate(ent):
        lines.append(f"  State {i}: {e:.4f}")
    lines.append("")

    lines.append("PERSISTENCE — DWELL TIMES (bars)")
    edt = expected_dwell_times(fitted.hmm.transmat_)
    emp = empirical_dwell_times(states)
    for i in range(fitted.n_states):
        emp_arr = emp.get(i, np.array([]))
        emp_mean = f"{emp_arr.mean():.1f}" if len(emp_arr) > 0 else "N/A"
        emp_med  = f"{float(np.median(emp_arr)):.1f}" if len(emp_arr) > 0 else "N/A"
        lines.append(
            f"  State {i}: expected={edt[i]:.1f}  "
            f"empirical_mean={emp_mean}  empirical_median={emp_med}"
        )
    lines.append("")

    lines.append("HEURISTIC LABELS")
    lines.append(f"  {HEURISTIC_DISCLAIMER}")
    for ls in sorted(labelled_states, key=lambda x: x.index):
        count = int((states == ls.index).sum())
        pct = 100.0 * count / len(states)
        lines.append(
            f"  State {ls.index}: {ls.label!r:<30s}"
            f"  return_rank={ls.rank_return}  vol_rank={ls.rank_volatility}"
            f"  ({count} bars, {pct:.1f}%)"
        )
    lines.append("")

    if scores_df is not None:
        lines.append("AIC / BIC SELECTION TABLE")
        lines.append(scores_df.to_string(index=False))
        lines.append("")

    # ── Stability section ────────────────────────────────────────────────────
    if stability_result is not None:
        n_runs = len(stability_result["models"])
        lines.append(f"STABILITY ANALYSIS  ({n_runs} independent runs)")
        lines.append(
            "  NOTE: ARI near 1.0 = consistent labelling; near 0 = unstable. "
            "State permutations across runs are normal and expected."
        )
        lines.append(f"  mean_ARI:     {stability_result['mean_ari']:.4f}")
        lines.append(f"  min_ARI:      {stability_result['min_ari']:.4f}")
        lines.append(f"  transmat_std: {stability_result['transmat_std']:.4f}"
                     "  (mean element-wise σ across runs)")
        lines.append(f"  means_std:    {stability_result['means_std']:.4f}")
        lines.append("")
        lines.append("  ARI matrix (pairwise run agreement):")
        mat = stability_result["ari_matrix"]
        header_row = "        " + "  ".join(f"r{j:02d}" for j in range(n_runs))
        lines.append(header_row)
        for i, row in enumerate(mat):
            vals = "  ".join(f"{v:.3f}" for v in row)
            lines.append(f"    r{i:02d}  {vals}")
        lines.append("")

    # ── Walk-forward section ─────────────────────────────────────────────────
    if wf_result is not None:
        labeled_mask = wf_result["regime"].notna()
        n_oos = int(labeled_mask.sum())
        n_total = len(wf_result)
        pct_oos = 100.0 * n_oos / n_total if n_total > 0 else 0.0
        first_oos = wf_result.index[labeled_mask][0]  if n_oos > 0 else "—"
        last_oos  = wf_result.index[labeled_mask][-1] if n_oos > 0 else "—"

        wf_window = getattr(wf_cfg, "walk_forward_train_window", "?")
        wf_recal  = getattr(wf_cfg, "walk_forward_recalibration", "?")
        lines.append(f"WALK-FORWARD VALIDATION  ({wf_recal}, train_window={wf_window})")
        lines.append(f"  OOS bars:    {n_oos:>7d}  ({pct_oos:.1f}% of total)")
        lines.append(f"  Warm-up:     {n_total - n_oos:>7d}  ({100.0 - pct_oos:.1f}%, no label)")
        lines.append(f"  First OOS:   {first_oos}")
        lines.append(f"  Last OOS:    {last_oos}")
        lines.append(f"\n  {HEURISTIC_DISCLAIMER}")
        lines.append("  OOS regime distribution:")
        wf_states = wf_result.loc[labeled_mask, "regime"].astype(int)
        label_list = [ls.label for ls in sorted(labelled_states, key=lambda x: x.index)]
        for s_idx in sorted(wf_states.unique()):
            c = int((wf_states == s_idx).sum())
            p = 100.0 * c / n_oos if n_oos > 0 else 0.0
            lbl = label_list[s_idx] if s_idx < len(label_list) else f"State {s_idx}"
            lines.append(f"    State {s_idx} {lbl!r:<28s}  {c:6d} bars  ({p:.1f}%)")
        lines.append("")

    # ── Regime analytics section ──────────────────────────────────────────────
    if regime_stats_df is not None:
        lines.append("REGIME PERFORMANCE ANALYTICS  (annualised, hourly bars @ 8760 h/yr)")
        lines.append(
            "  NOTE: In-sample statistics only. Past regime behaviour does not guarantee "
            "future returns. Log returns used throughout."
        )
        lines.append(regime_stats_df.to_string())
        lines.append("")

    if transition_table is not None:
        lines.append("FORWARD TRANSITION PROBABILITIES  (horizons: 1h, 6h, 1d, 3d, 1w)")
        lines.append(
            "  P(to_state | from_state, h bars ahead) = (A^h)[from, to]"
        )
        lines.append(transition_table.to_string(float_format=lambda x: f"{x:.3f}"))
        lines.append("")

    lines.append(sep)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote diagnostics → %s", path)


@click.group()
def main() -> None:
    """Regime Detection Engine — modular HMM-based market regime analysis."""


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config file.",
)
@click.option(
    "--n-states",
    "n_states_override",
    default=None,
    type=int,
    help="Override AIC/BIC selection with a fixed state count.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass the parquet data cache.",
)
@click.option(
    "--interactive",
    "save_interactive",
    is_flag=True,
    default=False,
    help="Also save interactive Plotly HTML plots alongside the static PNGs.",
)
@click.option(
    "--stability",
    "run_stability",
    is_flag=True,
    default=False,
    help="Run stability analysis (ARI across independent restarts). Overrides config.",
)
@click.option(
    "--stability-runs",
    "stability_runs_override",
    default=None,
    type=int,
    help="Number of independent stability runs (overrides config.evaluation.stability_n_runs).",
)
@click.option(
    "--walk-forward",
    "run_walk_forward",
    is_flag=True,
    default=False,
    help="Run walk-forward validation harness. Overrides config.",
)
@click.option(
    "--wf-window",
    "wf_window_override",
    default=None,
    type=str,
    help="Training window for walk-forward, e.g. '180d' (overrides config).",
)
@click.option(
    "--emission-dist",
    "emission_dist_override",
    default=None,
    type=click.Choice(["gaussian", "student_t"]),
    help="Emission distribution: 'gaussian' (default) or 'student_t' (fat-tailed ECM). "
         "Overrides config.model.emission_dist.",
)
def run(
    config_path: str,
    n_states_override: int | None,
    no_cache: bool,
    save_interactive: bool,
    run_stability: bool,
    stability_runs_override: int | None,
    run_walk_forward: bool,
    wf_window_override: str | None,
    emission_dist_override: str | None,
) -> None:
    """Run the full regime detection pipeline for one asset config."""
    # ── 1. Config ──────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    symbol = cfg.asset.symbol

    # CLI flags override config evaluation settings
    do_stability    = run_stability    or cfg.evaluation.run_stability
    do_walk_forward = run_walk_forward or cfg.evaluation.run_walk_forward
    stability_n_runs = stability_runs_override or cfg.evaluation.stability_n_runs
    wf_window = wf_window_override or cfg.evaluation.walk_forward_train_window
    emission_dist = emission_dist_override or cfg.model.emission_dist

    click.echo(f"\n{'═'*60}")
    click.echo(f"  Regime Detection Engine  ·  {symbol}")
    click.echo(f"{'═'*60}")

    # ── 2. Data ────────────────────────────────────────────────────────────
    cache_dir = Path("results/cache") if not no_cache else None
    source = YFinanceSource(cache_dir=cache_dir)
    click.echo(
        f"  [1/8] Loading data  "
        f"period={cfg.asset.period}  interval={cfg.asset.interval}"
    )
    df = source.load(cfg.asset.symbol, cfg.asset.period, cfg.asset.interval)
    click.echo(f"        {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── 3. Features ────────────────────────────────────────────────────────
    click.echo("  [2/8] Building features ...")
    pipeline = _build_pipeline(cfg.features)
    df_features = pipeline.transform(df)
    click.echo(
        f"        {pipeline.output_columns}  →  {len(df_features)} rows after dropna"
    )

    feature_cols = pipeline.output_columns
    X = df_features[feature_cols].values

    train_kwargs: dict = dict(
        n_restarts=cfg.model.n_restarts,
        covariance_type=cfg.model.covariance_type,
        n_iter=cfg.model.n_iter,
        init_strategy=cfg.model.init_strategy,
        seed_base=cfg.model.seed_base,
        feature_names=feature_cols,
        emission_dist=emission_dist,
    )

    # ── 4. Train / select ──────────────────────────────────────────────────
    scores_df: pd.DataFrame | None = None
    if n_states_override is not None:
        click.echo(f"  [3/8] Training HMM  n_states={n_states_override} (fixed) ...")
        fitted = train_hmm(X, n_states_override, **train_kwargs)
    else:
        click.echo(
            f"  [3/8] AIC/BIC selection over {cfg.model.candidate_states} states "
            f"(criterion={cfg.selection.criterion.upper()}) ..."
        )
        fitted, scores_df = select_n_states(
            X,
            cfg.model.candidate_states,
            criterion=cfg.selection.criterion,
            **train_kwargs,
        )
        click.echo(f"        → selected n_states={fitted.n_states}")

    click.echo(
        f"        log-lik={fitted.log_likelihood:.2f}  "
        f"AIC={fitted.aic:.2f}  BIC={fitted.bic:.2f}  seed={fitted.seed}"
    )

    # ── 5. Inference ───────────────────────────────────────────────────────
    click.echo("  [4/8] Decoding regimes (Viterbi + forward-backward) ...")
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    posteriors = forward_backward_posteriors(fitted.hmm, X_scaled)

    # ── 6. Evaluation ──────────────────────────────────────────────────────
    click.echo("  [5/8] Running evaluation ...")
    transmat = fitted.hmm.transmat_
    stationary = stationary_distribution(transmat)

    stability_result: dict | None = None
    if do_stability:
        click.echo(
            f"        Stability analysis: {stability_n_runs} runs × "
            f"{cfg.model.n_restarts} restarts ..."
        )
        stability_result = stability_across_restarts(
            X, fitted.n_states, stability_n_runs,
            n_restarts=cfg.model.n_restarts,
            covariance_type=cfg.model.covariance_type,
            n_iter=cfg.model.n_iter,
            seed_base=cfg.model.seed_base,
        )
        click.echo(
            f"        mean_ARI={stability_result['mean_ari']:.4f}  "
            f"min_ARI={stability_result['min_ari']:.4f}  "
            f"transmat_std={stability_result['transmat_std']:.4f}"
        )

    wf_result: pd.DataFrame | None = None
    if do_walk_forward:
        click.echo(
            f"        Walk-forward: {cfg.evaluation.walk_forward_recalibration} "
            f"recalibration, train_window={wf_window} ..."
        )
        harness = WalkForwardHarness(
            recalibration=cfg.evaluation.walk_forward_recalibration,
            train_window=wf_window,
        )
        wf_kwargs = dict(
            n_restarts=cfg.evaluation.walk_forward_n_restarts,
            covariance_type=cfg.model.covariance_type,
            n_iter=cfg.model.n_iter,
            seed_base=cfg.model.seed_base,
        )
        wf_result = harness.run(
            df_features, fitted.n_states,
            feature_cols=feature_cols,
            **wf_kwargs,
        )
        n_oos = int(wf_result["regime"].notna().sum())
        pct   = 100.0 * n_oos / len(wf_result)
        click.echo(f"        OOS bars labelled: {n_oos} / {len(wf_result)} ({pct:.1f}%)")

    # ── 7. Labels + regime analytics ──────────────────────────────────────
    click.echo("  [6/8] Labelling states ...")
    labelled_states = rank_states(fitted)
    label_list = [ls.label for ls in sorted(labelled_states, key=lambda x: x.index)]

    return_col = fitted.feature_names[0]  # log_return is always first
    regime_stats = compute_regime_stats(
        returns=df_features[return_col].values,
        states=states,
        labels=label_list,
    )
    regime_stats_df = regime_stats_to_dataframe(regime_stats)
    transition_tbl = regime_transition_table(
        fitted.hmm.transmat_,
        labels=label_list,
    )

    # ── 8. Plots ───────────────────────────────────────────────────────────
    output_dir = Path(cfg.run.output_dir.format(symbol=symbol))
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"  [7/8] Generating plots → {plots_dir} ...")
    _plot_specs = [
        ("price_with_regimes",
         plot_price_with_regimes(df_features, states, labels=label_list, symbol=symbol),
         iplot_price_with_regimes(df_features, states, labels=label_list, symbol=symbol)),
        ("transition_heatmap",
         plot_transition_heatmap(transmat, labels=label_list, symbol=symbol),
         iplot_transition_heatmap(transmat, labels=label_list, symbol=symbol)),
        ("regime_timeline",
         plot_regime_timeline(df_features, states, labels=label_list, symbol=symbol),
         iplot_regime_timeline(df_features, states, labels=label_list, symbol=symbol)),
        ("per_regime_returns",
         plot_per_regime_returns(df_features, states, labels=label_list, symbol=symbol),
         iplot_per_regime_returns(df_features, states, labels=label_list, symbol=symbol)),
    ]
    for name, static_fig, interactive_fig in _plot_specs:
        static_fig.savefig(plots_dir / f"{name}.png", dpi=150)
        plt.close(static_fig)
        if save_interactive:
            interactive_fig.write_html(str(plots_dir / f"{name}.html"))

    if save_interactive:
        click.echo(f"        Interactive HTML plots saved to {plots_dir}")

    # ── 9. Diagnostics ─────────────────────────────────────────────────────
    diag_path = output_dir / "diagnostics.txt"
    _write_diagnostics(
        diag_path,
        symbol=symbol,
        fitted=fitted,
        states=states,
        df_features=df_features,
        labelled_states=labelled_states,
        scores_df=scores_df,
        stationary=stationary,
        stability_result=stability_result,
        wf_result=wf_result,
        wf_cfg=cfg.evaluation,
        regime_stats_df=regime_stats_df,
        transition_table=transition_tbl,
    )

    # ── 10. Regimes parquet ────────────────────────────────────────────────
    click.echo("  [8/8] Writing outputs ...")
    df_regimes = df_features.copy()
    df_regimes["regime"] = states
    df_regimes["regime_label"] = [label_list[int(s)] for s in states]
    for i in range(fitted.n_states):
        df_regimes[f"regime_proba_{i}"] = posteriors[:, i]

    regimes_path = output_dir / "regimes.parquet"
    df_regimes.to_parquet(regimes_path)

    analytics_path = output_dir / "regime_analytics.parquet"
    regime_stats_df.to_parquet(analytics_path)

    click.echo(f"        regimes.parquet       → {regimes_path}")
    click.echo(f"        regime_analytics.parquet → {analytics_path}")
    click.echo(f"        diagnostics.txt       → {diag_path}")

    if wf_result is not None:
        wf_path = output_dir / "walk_forward_regimes.parquet"
        wf_result.to_parquet(wf_path)
        click.echo(f"        walk_forward_regimes  → {wf_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    click.echo(f"\n{'═'*60}")
    click.echo("  SUMMARY")
    click.echo(f"  Symbol:       {symbol}")
    click.echo(f"  n_states:     {fitted.n_states}  (seed={fitted.seed})")
    click.echo(f"  Log-lik:      {fitted.log_likelihood:.2f}")
    click.echo(f"  AIC / BIC:    {fitted.aic:.2f} / {fitted.bic:.2f}")
    click.echo(f"  Observations: {len(df_features)}")
    click.echo(f"  Output:       {output_dir}")
    if stability_result is not None:
        click.echo(
            f"  Stability:    mean_ARI={stability_result['mean_ari']:.4f}  "
            f"min_ARI={stability_result['min_ari']:.4f}"
        )
    if wf_result is not None:
        n_oos = int(wf_result["regime"].notna().sum())
        click.echo(f"  Walk-forward: {n_oos} OOS bars labelled")
    click.echo(f"\n  {HEURISTIC_DISCLAIMER}")
    click.echo("")
    click.echo(
        f"  {'State':<8} {'Label':<28} {'Bars':>6}  {'Wt%':>5}  "
        f"{'RetAnn%':>8}  {'VolAnn%':>8}  {'Sharpe':>7}  {'MaxDD%':>7}"
    )
    click.echo("  " + "-" * 82)
    for rs in regime_stats:
        sharpe_str = f"{rs.sharpe_ann:>7.2f}" if not np.isnan(rs.sharpe_ann) else "    N/A"
        click.echo(
            f"  {rs.state:<8} {rs.label:<28} {rs.count:>6d}  {100*rs.weight:>5.1f}%"
            f"  {100*rs.mean_return_ann:>+8.2f}  {100*rs.vol_ann:>8.2f}"
            f"  {sharpe_str}  {100*rs.max_drawdown:>7.2f}"
        )
    click.echo(f"{'═'*60}\n")
