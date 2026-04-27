# Regime Detection Engine

A modular, general-purpose **Market Regime Detection Engine** built around Hidden Markov Models. The system ingests financial time series, fits Gaussian HMMs with automatic state-count selection via AIC/BIC, decodes market regimes with Viterbi, and produces interpretable diagnostics and visualisations.

**Bitcoin (BTC-USD, hourly, 730 days) is the baseline asset**, but the engine is designed to run on any asset via a YAML config — no code changes required.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast, lockfile-based package manager

---

## Installation

```bash
git clone <repo-url> && cd regime-detection-engine
uv sync
```

---

## Quick start

```bash
# Run the BTC-USD baseline
uv run rde run --config configs/btc.yaml

# Run on ETH or SPY
uv run rde run --config configs/eth.yaml
uv run rde run --config configs/spy.yaml
```

Outputs are written to `results/<SYMBOL>/`:

| File | Contents |
|------|----------|
| `diagnostics.txt` | Transition matrix, state means, persistence, heuristic labels with disclaimer, AIC/BIC table |
| `regimes.parquet` | Full labelled DataFrame: `regime`, `regime_label`, `regime_proba_*` columns |
| `plots/price_with_regimes.png` | Close price coloured by decoded regime |
| `plots/transition_heatmap.png` | Annotated transition probability heatmap |
| `plots/regime_timeline.png` | Horizontal regime strip |
| `plots/per_regime_returns.png` | Return distributions conditional on regime |

---

## CLI reference

```
Usage: rde run [OPTIONS]

Options:
  --config PATH              Path to YAML config file.  [required]
  --n-states INTEGER         Override AIC/BIC selection with a fixed state count.
  --no-cache                 Bypass the parquet data cache.
  --interactive              Also save interactive Plotly HTML plots alongside static PNGs.
  --stability                Run stability analysis (ARI across independent restarts).
  --stability-runs INTEGER   Number of independent stability runs.
  --walk-forward             Run walk-forward validation harness.
  --wf-window TEXT           Training window for walk-forward, e.g. '180d'.
```

### Examples

```bash
# Force 4 states instead of AIC/BIC selection
uv run rde run --config configs/btc.yaml --n-states 4

# Skip the data cache (re-download from yfinance)
uv run rde run --config configs/btc.yaml --no-cache

# Save interactive HTML plots alongside PNGs
uv run rde run --config configs/btc.yaml --interactive

# Run stability analysis (ARI across 5 independent restarts)
uv run rde run --config configs/btc.yaml --stability --stability-runs 5

# Run walk-forward validation with a custom training window
uv run rde run --config configs/btc.yaml --walk-forward --wf-window 90d
```

---

## Configuration

Each asset is described by a YAML file. Example:

```yaml
asset:
  symbol: BTC-USD
  period: 730d     # max for 1h interval; do not use start/end
  interval: 1h

features:
  - name: LogReturns
  - name: RollingVolatility
    params:
      window: 24
  - name: SmoothedReturns
    params:
      window: 12

model:
  candidate_states: [2, 3, 4, 5, 6]   # AIC/BIC selects the best
  covariance_type: full
  n_restarts: 10                        # multiple restarts, keeps best log-lik
  n_iter: 1000
  init_strategy: kmeans
  seed_base: 42

selection:
  criterion: bic   # or "aic"

run:
  output_dir: results/{symbol}/

evaluation:
  run_stability: false       # enable to add ARI analysis to diagnostics
  stability_n_runs: 3
  run_walk_forward: false    # enable for out-of-sample labelling
  walk_forward_train_window: 180d
  walk_forward_recalibration: monthly
  walk_forward_n_restarts: 3
```

To add a new asset, copy any existing config and change `symbol`. No Python code changes are needed.

---

## Architecture

```
src/rde/
├── data/          DataSource ABC + yfinance source + parquet cache
├── features/      FeatureTransformer ABC + pipeline + LogReturns, RollingVol, SmoothedReturns
├── models/        GaussianHMM training (multi-restart) + AIC/BIC selection
├── inference/     Viterbi decode, forward log-likelihood, forward-backward posteriors
├── evaluation/    Persistence (dwell times), stability (ARI), transition entropy, walk-forward
├── viz/           Static (matplotlib) + interactive (Plotly) plots
├── labeling/      Heuristic state ranking by return and volatility
├── config/        YAML schema dataclasses + validated loader
└── cli.py         Click entry point — wires all modules together
```

**Non-negotiable design rules** (see `CLAUDE.md` §2 for the full list):

- The number of regimes is **never hardcoded** — AIC/BIC selects it.
- Training **always uses multiple random restarts** and keeps the best log-likelihood.
- yfinance is called with `period="730d", interval="1h"` — never with `start`/`end`.
- Regime labels are **heuristic interpretations**, not ground truth. Every output carrying a label includes an explicit disclaimer.
- Walk-forward validation uses **no future data** — each bar is labelled by a model trained strictly before it.

---

## Tests

```bash
uv run pytest
```

174 tests cover inference math (Jurafsky ice-cream HMM as ground truth), feature transformers, model selection, all four visualisations (static + interactive), config validation, stability analysis, walk-forward harness, and full CLI integration.

---

## Regime labels disclaimer

> Labels such as `bearish`, `bullish`, `low_return_high_vol`, etc. are heuristic
> interpretations of the model's state means and **do not represent ground truth
> market conditions**. They are produced by ranking states on their mean log-return
> and volatility. Use them as exploration aids, not trading signals.

---

## Version

Current release: **v1.0.0**
