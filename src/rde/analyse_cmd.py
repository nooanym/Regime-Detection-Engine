"""``rde analyse`` — run the Phase 22–30 analysis pipeline on a fitted model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FEATURE_REGISTRY = {
    "LogReturns": lambda p: _lazy_feature("LogReturns", p),
    "RollingVolatility": lambda p: _lazy_feature("RollingVolatility", p),
    "SmoothedReturns": lambda p: _lazy_feature("SmoothedReturns", p),
    "LogVolume": lambda p: _lazy_feature("LogVolume", p),
    "RollingVolumeZScore": lambda p: _lazy_feature("RollingVolumeZScore", p),
}


def _lazy_feature(name: str, params: dict):
    """Instantiate a feature transformer by name, passing params as kwargs."""
    from rde.features.returns import LogReturns, SmoothedReturns
    from rde.features.volatility import RollingVolatility
    from rde.features.volume import LogVolume, RollingVolumeZScore

    _constructors = {
        "LogReturns": lambda p: LogReturns(),
        "RollingVolatility": lambda p: RollingVolatility(**p),
        "SmoothedReturns": lambda p: SmoothedReturns(**p),
        "LogVolume": lambda p: LogVolume(),
        "RollingVolumeZScore": lambda p: RollingVolumeZScore(**p),
    }
    return _constructors[name](params)


@click.command("analyse")
@click.option(
    "--config", "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to asset YAML config (e.g. configs/btc.yaml).",
)
@click.option(
    "--output-dir", "output_dir",
    default=None,
    type=click.Path(),
    help="Where to write analysis_report.json and analysis.md. "
         "Defaults to results/<symbol>/analysis/.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Bypass parquet cache when loading data.",
)
@click.option(
    "--n-states",
    default=None,
    type=int,
    help="Override AIC/BIC model selection with a fixed state count.",
)
def analyse_cmd(
    config_path: str,
    output_dir: str | None,
    no_cache: bool,
    n_states: int | None,
) -> None:
    """Run the full Phase 22–30 analysis suite on a model trained from --config."""

    # 1. Load config
    from rde.config.loader import load_config

    cfg = load_config(config_path)
    symbol = cfg.asset.symbol

    # 2. Load data
    from rde.data.yfinance_source import YFinanceSource

    cache_dir = None if no_cache else Path("results/cache")
    src = YFinanceSource(cache_dir=cache_dir)
    df_raw = src.load(symbol, cfg.asset.period, cfg.asset.interval)
    click.echo(f"[analyse] Loaded {len(df_raw)} bars for {symbol}")

    # 3. Build features
    from rde.features.pipeline import FeaturePipeline

    transformers = []
    for fc in cfg.features:
        if fc.name not in _FEATURE_REGISTRY:
            click.echo(f"Unknown feature: {fc.name!r}", err=True)
            sys.exit(1)
        transformers.append(_FEATURE_REGISTRY[fc.name](fc.params))
    pipeline = FeaturePipeline(transformers)
    df_features = pipeline.transform(df_raw)
    feature_names = pipeline.output_columns
    X_raw = df_features[feature_names].values

    # 4. Scale features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 5. Train model (or use fixed n_states)
    from rde.models.hmm import train_hmm
    from rde.models.selection import select_n_states

    if n_states is not None:
        fitted = train_hmm(
            X,
            n_states,
            n_restarts=cfg.model.n_restarts,
            n_iter=cfg.model.n_iter,
            seed_base=cfg.model.seed_base,
        )
        click.echo(f"[analyse] Trained HMM with {n_states} states")
    else:
        fitted, scores_df = select_n_states(
            X,
            candidate_states=cfg.model.candidate_states,
            criterion=cfg.selection.criterion,
            n_restarts=cfg.model.n_restarts,
            n_iter=cfg.model.n_iter,
            seed_base=cfg.model.seed_base,
        )
        click.echo(
            f"[analyse] Selected {fitted.n_states} states by "
            f"{cfg.selection.criterion.upper()}"
        )

    # 6. Decode
    from rde.inference.viterbi import viterbi_decode
    from rde.inference.smoothing import forward_backward_posteriors

    regimes = viterbi_decode(fitted.hmm, X)
    posteriors = forward_backward_posteriors(fitted.hmm, X)
    transmat = fitted.hmm.transmat_

    # 7. Build returns series (use first feature column = log returns)
    returns = pd.Series(X_raw[:, 0], index=df_features.index)

    # 8. Determine output dir
    if output_dir is None:
        out_path = Path(f"results/{symbol}/analysis")
    else:
        out_path = Path(output_dir)

    # 9. Run analysis pipeline
    from rde.analysis.pipeline import AnalysisPipeline, AnalysisConfig
    from rde.analysis.reporting import save_report

    analysis_cfg = AnalysisConfig(symbol=symbol, returns_col_idx=0)
    ap = AnalysisPipeline(config=analysis_cfg)

    click.echo("[analyse] Running analysis modules...")
    report = ap.run(
        X=X,
        posteriors=posteriors,
        transmat=transmat,
        returns=returns,
        regimes=regimes,
        feature_names=feature_names,
    )

    # 10. Save
    save_report(report, out_path, symbol)
    click.echo(f"[analyse] Report written to {out_path}/")
    click.echo(f"[analyse] JSON:     {out_path}/analysis_report.json")
    click.echo(f"[analyse] Markdown: {out_path}/analysis.md")
