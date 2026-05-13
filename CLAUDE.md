# CLAUDE.md — Regime Detection Engine


> This file is read by Claude Code on every session. It is the single source of truth for what we are building, how, and why. **Update this file whenever architectural decisions change.** The Notion workspace at `Regime Detection Engine` holds living project management; this file holds invariants for the codebase.

---

## 1. What we are building

A general-purpose, modular **Market Regime Detection Engine** built around Hidden Markov Models. The system ingests financial time series, fits HMMs with automatic state-count selection, decodes market regimes, and produces interpretable diagnostics and visualizations.

**Bitcoin (BTC-USD, hourly, 730 days) is the mandatory baseline asset**, but the entire system is architected from day one to extend cleanly to other cryptocurrencies, equities, and indices via configuration files. No code changes should be required to run on a new asset.

This is **not** a one-off Bitcoin script. It is a research engine.

---

## 2. Non-negotiable rules

1. **Never hardcode the number of regimes.** Use AIC/BIC selection across a candidate set.
2. **Never single-fit.** HMM training (Baum-Welch) is non-convex. Always do multiple random restarts and select the best by log-likelihood.
3. **Never use `start`/`end` parameters with yfinance.** The mandate is `period="730d"`, `interval="1h"`. Yahoo's intraday retention cap is ~730 days; using `start`/`end` is an antipattern that breaks silently.
4. **Treat regime labels as interpretations, not facts.** Every diagnostics output containing `bullish`, `bearish`, `high_vol`, etc. carries an explicit "heuristic, not ground truth" disclaimer.
5. **Never leak future data in validation.** Walk-forward only, with parameters locked before the test window.
6. **No tight coupling between layers.** Data, features, models, inference, evaluation, viz, labeling are independent modules with stable interfaces.
7. **No Bitcoin-specific code paths in the engine itself.** Asset-specific behavior lives in YAML config files, never in Python source.
8. **Reproducibility.** Every run sets a seed, logs it, and produces deterministic outputs given the same config + data.

---

## 3. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Modern typing, performance |
| Dep management | `uv` | Faster, lockfile-based, reproducible |
| HMM | `hmmlearn` | Industry standard, log-space internals, tested |
| Data | `yfinance` | Required by spec; pluggable behind a `DataSource` ABC |
| Numerics | `numpy`, `pandas` | Standard |
| Scaling / kmeans init | `scikit-learn` | Already a `hmmlearn` transitive dep |
| Static plots | `matplotlib` | Reports, embeddable PNG |
| Interactive plots | `plotly` | Notebook exploration |
| Config | `pyyaml` + dataclasses (or `pydantic`) | Typed, validated |
| Cache | `pyarrow` (parquet) | Fast, schema-preserving |
| CLI | `click` | Standard, ergonomic |
| Tests | `pytest` | Standard |
| Lint/format | `ruff` | Fast, batteries-included |

**Forbidden additions without an entry in the Decision Log:** new data sources, new ML libraries, new plotting libraries.

---

## 4. Repository layout

```
regime-detection-engine/
├── src/rde/
│   ├── __init__.py              # exposes __version__
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py              # DataSource ABC
│   │   ├── yfinance_source.py   # period="730d", interval="1h"
│   │   └── cache.py             # parquet read/write
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py              # FeatureTransformer ABC
│   │   ├── returns.py           # log_returns, smoothed_returns
│   │   ├── volatility.py        # rolling_volatility
│   │   ├── momentum.py          # placeholder for future
│   │   └── pipeline.py          # FeaturePipeline composer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hmm.py               # GaussianHMM wrapper
│   │   ├── selection.py         # AIC/BIC across n_states
│   │   └── init_strategies.py   # kmeans / random / equal-prior
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── forward.py           # log-likelihood
│   │   ├── viterbi.py           # MAP decoding
│   │   └── smoothing.py         # forward-backward posteriors
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── persistence.py       # dwell-time analysis
│   │   ├── stability.py         # ARI across restarts
│   │   ├── transition.py        # stationary distribution, entropy
│   │   └── walk_forward.py      # rolling re-fit harness
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── static_plots.py      # matplotlib
│   │   └── interactive.py       # plotly
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── ranking.py           # rank by mean / vol
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py            # dataclasses
│   │   └── loader.py            # YAML loader + validation
│   └── cli.py                   # `rde run --config <path>`
├── configs/
│   ├── btc.yaml
│   ├── eth.yaml
│   └── spy.yaml
├── notebooks/
│   ├── 01_btc_baseline.ipynb
│   └── 02_model_selection.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_inference.py        # uses Jurafsky ice-cream HMM as ground truth
│   └── test_selection.py
├── results/                     # gitignored except .gitkeep
│   └── cache/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── LICENSE
├── README.md
└── CLAUDE.md                    # this file
```

---

## 5. Module specifications

### 5.1 `rde.data`

**Responsibility:** load OHLCV data for any (symbol, period, interval). Cache locally to avoid re-hitting yfinance.

**Public API:**

```python
class DataSource(ABC):
    @abstractmethod
    def load(self, symbol: str, period: str, interval: str) -> pd.DataFrame: ...

class YFinanceSource(DataSource):
    def __init__(self, cache_dir: Path | None = None): ...
    def load(self, symbol: str, period: str, interval: str) -> pd.DataFrame: ...
```

**Returned DataFrame schema:** index = tz-aware DatetimeIndex; columns ⊇ `{Open, High, Low, Close, Volume}`.

**Mandatory behavior:**

- For BTC-USD baseline, `load("BTC-USD", "730d", "1h")` must work.
- Use `yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)`.
- `auto_adjust=True` is the modern yfinance default; do not depend on `Adj Close` existing as a separate column.
- Cache responses to `<cache_dir>/<symbol>_<period>_<interval>.parquet`. Cache hit if file exists and is younger than 1 hour (configurable).
- Drop rows with NaN in `Close` before returning.

**Pitfalls to handle:**

- yfinance occasionally returns a multi-level column index when called with a list of symbols. We always pass a single symbol, but flatten defensively just in case.
- Hourly data has a hard ~730-day retention; using `start`/`end` outside that window silently returns empty data. Reject any caller-provided `period` longer than `730d` for `interval="1h"` with a clear error.

### 5.2 `rde.features`

**Responsibility:** transform a price DataFrame into a feature matrix suitable for HMM input.

**Public API:**

```python
class FeatureTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @property
    @abstractmethod
    def output_columns(self) -> list[str]: ...

class FeaturePipeline:
    def __init__(self, transformers: list[FeatureTransformer]): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    @property
    def output_columns(self) -> list[str]: ...
```

**Baseline transformers:**

- `LogReturns`: `log(Close / Close.shift(1))`. Output column: `log_return`.
- `RollingVolatility(window: int = 24)`: `log_return.rolling(window).std()`. Output: `volatility_w{window}`.
- `SmoothedReturns(window: int = 12)`: `log_return.rolling(window).mean()`. Output: `smoothed_return_w{window}`.

**Rules:**

- Each transformer is **pure**: input DataFrame, return new DataFrame with added columns.
- The pipeline always `dropna()` at the end (rolling windows produce leading NaNs).
- The pipeline is **the only place** features are produced. Models, inference, and evaluation do not compute features themselves.
- Standardization (StandardScaler from sklearn) happens in the model layer, not the feature layer, so the same features can feed scaled or unscaled consumers.

### 5.3 `rde.models`

**Responsibility:** train Gaussian HMMs robustly. Select the best `n_states` by AIC/BIC.

**Public API:**

```python
@dataclass
class FittedModel:
    hmm: GaussianHMM            # the hmmlearn estimator
    scaler: StandardScaler
    n_states: int
    log_likelihood: float
    aic: float
    bic: float
    feature_names: list[str]
    seed: int                    # the seed of the winning restart
    all_restart_scores: list[float]  # for diagnostics

def train_hmm(
    X: np.ndarray,
    n_states: int,
    *,
    n_restarts: int = 10,
    covariance_type: str = "full",
    n_iter: int = 1000,
    init_strategy: str = "kmeans",
    seed_base: int = 42,
) -> FittedModel: ...

def select_n_states(
    X: np.ndarray,
    candidate_states: list[int],
    *,
    criterion: str = "bic",          # "aic" or "bic"
    **train_kwargs,
) -> tuple[FittedModel, pd.DataFrame]:  # (chosen_model, scores_table)
    ...
```

**Implementation rules:**

- Standardize `X` with `StandardScaler` before passing to HMM. Persist the scaler with the model so inference uses the same transform.
- For each restart `i in range(n_restarts)`, set `random_state = seed_base + i`, fit, score on the training data via `model.score(X)`, keep the best.
- Compute AIC and BIC manually after fitting:
  - `n_params = n_states * (n_states - 1)            # transitions, free`
                `+ (n_states - 1)                    # initial probs, free`
                `+ n_states * n_features              # means`
                `+ n_states * n_features * (n_features + 1) / 2  # full cov; adjust for diagonal`
  - `AIC = 2 * n_params - 2 * log_likelihood`
  - `BIC = n_params * log(T) - 2 * log_likelihood` where `T = len(X)`
- Verify against `hmmlearn`'s built-in `aic`/`bic` properties (available in recent versions) — they should match.
- If a restart fails to converge (catch `ConvergenceWarning`), drop it from the pool but log it. If all restarts fail, raise.
- `select_n_states` returns the model with the **lowest** chosen criterion plus a DataFrame with one row per `n_states` and columns `[n_states, log_likelihood, aic, bic, n_params, converged_restarts]`.

### 5.4 `rde.inference`

**Responsibility:** the three canonical HMM inference algorithms. Reusable across models.

**Public API:**

```python
def forward_log_likelihood(model: GaussianHMM, X: np.ndarray) -> float: ...
def viterbi_decode(model: GaussianHMM, X: np.ndarray) -> np.ndarray: ...   # int array shape (T,)
def forward_backward_posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:  # (T, n_states)
    ...
```

These are thin wrappers over `model.score()`, `model.predict()`, and `model.predict_proba()` respectively, but they exist as a separate module so we can swap in a custom implementation later without touching downstream code.

**Test fixture (mandatory):** the Jurafsky ice-cream HMM (Appendix A, Speech and Language Processing).

- States: `Hot`, `Cold`. Observations: ice-creams eaten ∈ {1, 2, 3}.
- π = [0.8, 0.2]. A = [[0.6, 0.4], [0.5, 0.5]]. B[Hot] = [0.2, 0.4, 0.4]. B[Cold] = [0.5, 0.4, 0.1].
- Observation sequence `3 1 3` has known forward log-likelihood and Viterbi path. Tests must reproduce these.
- Note: the Jurafsky example uses categorical (multinomial) emissions, so the test will be against a `MultinomialHMM` or `CategoricalHMM`. We test the inference *algorithms* with a categorical model in tests, even though our production models are Gaussian. The point is to verify the math, not the emission distribution.

### 5.5 `rde.evaluation`

**Responsibility:** quantitative regime diagnostics that don't require visual inspection.

**Public API:**

```python
def expected_dwell_times(transmat: np.ndarray) -> np.ndarray: ...
    # 1 / (1 - p_ii) per state

def empirical_dwell_times(states: np.ndarray) -> dict[int, np.ndarray]: ...
    # Per-state arrays of run lengths

def stationary_distribution(transmat: np.ndarray) -> np.ndarray: ...
    # Left eigenvector for eigenvalue 1

def transition_entropy(transmat: np.ndarray) -> np.ndarray: ...
    # Per-state Shannon entropy of outgoing transitions

def stability_across_restarts(
    X: np.ndarray, n_states: int, n_runs: int, **train_kwargs,
) -> dict:
    # Train n_runs models with different seed_base values.
    # Return: ARI matrix between all pairs, parameter dispersion stats.

class WalkForwardHarness:
    def __init__(self, recalibration: str = "monthly", train_window: str = "180d"): ...
    def run(self, df_features: pd.DataFrame, n_states: int, **train_kwargs) -> pd.DataFrame:
        # Returns a DataFrame indexed like df_features with columns [regime, regime_proba_*]
        # filled only on out-of-sample bars. No future leakage.
```

**Walk-forward correctness rules:**

- At each recalibration boundary, fit using only data **strictly before** the boundary.
- The fitted model produces regime labels for the period from the boundary up to the next recalibration.
- A bar's regime is set by the model whose training window ended before that bar. Period.
- v1.0 does not need a full backtest. The harness emits a labelled DataFrame; downstream PnL is post-v1.0.

### 5.6 `rde.viz`

**Responsibility:** make the regimes legible. Static for reports, interactive for exploration.

**Required plots in v1.0:**

- `plot_price_with_regimes(df, states, labels=None) -> Figure`: line chart of `Close` colored by state. One color per state; legend with labels (heuristic if provided, else `State 0`, `State 1`, ...).
- `plot_transition_heatmap(transmat, labels=None) -> Figure`: annotated heatmap of the transition matrix.
- `plot_regime_timeline(df, states, labels=None) -> Figure`: a horizontal strip showing regime over time, useful as a small companion below the price plot.
- `plot_per_regime_returns(df, states, return_col="log_return") -> Figure`: histograms of returns conditional on regime, on shared axes.

Each of the above has a static (matplotlib) and interactive (plotly) variant. Tests verify the function returns a `Figure` object and does not raise on empty edge cases.

**Style rules:**

- Color palette is deterministic per state index (use `matplotlib.cm.tab10` for ≤10 states).
- Always include a title indicating the asset and date range.
- Always set `plt.tight_layout()` before returning.
- Never call `plt.show()` inside the library. Saving and displaying are caller responsibilities.

### 5.7 `rde.labeling`

**Responsibility:** assign human-readable labels to abstract state indices. **Always heuristic.**

**Public API:**

```python
@dataclass
class LabelledState:
    index: int
    label: str
    rank_return: int    # 0 = lowest mean return
    rank_volatility: int  # 0 = lowest volatility

def rank_states(model: FittedModel, return_feature: str = "log_return", vol_feature: str | None = "volatility_w24") -> list[LabelledState]: ...
```

**Labelling logic (v1.0):**

- Sort states by mean of the `return_feature`. The lowest gets `rank_return=0`, etc.
- If `vol_feature` is given, also sort by volatility-feature mean.
- Compose labels from the two ranks, e.g., `low_return_high_vol`, `mid_return_mid_vol`. With two states, the labels can degenerate to `bearish` / `bullish`. Whatever the outcome, **the diagnostics output prepends a notice**: "Labels are heuristic interpretations of the model's state means and do not represent ground truth market conditions."

### 5.8 `rde.config`

**Responsibility:** load and validate YAML configs into typed dataclasses.

**Schema (Python dataclasses):**

```python
@dataclass
class AssetConfig:
    symbol: str
    period: str = "730d"
    interval: str = "1h"

@dataclass
class FeatureConfig:
    name: str
    params: dict = field(default_factory=dict)

@dataclass
class ModelConfig:
    candidate_states: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    covariance_type: str = "full"
    n_restarts: int = 10
    n_iter: int = 1000
    init_strategy: str = "kmeans"
    seed_base: int = 42

@dataclass
class SelectionConfig:
    criterion: str = "bic"

@dataclass
class RunConfig:
    output_dir: str = "results/{symbol}/"

@dataclass
class Config:
    asset: AssetConfig
    features: list[FeatureConfig]
    model: ModelConfig
    selection: SelectionConfig
    run: RunConfig
```

**Validation:**

- `criterion` must be `"aic"` or `"bic"`.
- `covariance_type` must be one of `"full"`, `"diag"`, `"tied"`, `"spherical"`.
- All `candidate_states` ≥ 2.
- For `interval="1h"`, reject `period` longer than `"730d"`.

### 5.9 `rde.cli`

**Responsibility:** one command, one config, one run.

```
$ uv run rde run --config configs/btc.yaml
$ uv run rde run --config configs/btc.yaml --n-states 4    # override selection
$ uv run rde run --config configs/btc.yaml --no-cache       # bypass parquet cache
```

**Pipeline (in order):**

1. Load config.
2. Load data via `DataSource`.
3. Apply `FeaturePipeline`.
4. Standardize and call `select_n_states` (or `train_hmm` if `--n-states` is given).
5. Decode regimes via Viterbi; compute posteriors via forward-backward.
6. Run evaluation (persistence, stability, walk-forward).
7. Generate labels via `rank_states`.
8. Produce static plots and write to `results/<symbol>/plots/`.
9. Write `results/<symbol>/diagnostics.txt` with transition matrix, state stats, persistence, labels (with heuristic disclaimer), and AIC/BIC table.
10. Write `results/<symbol>/regimes.parquet` (the labelled DataFrame).
11. Print a one-screen summary to stdout.

---

## 6. Coding standards

- **Type hints required** on every public function and method.
- **Docstrings** on every public function in NumPy or Google style. One-line summary, then params, then returns, then notes (especially numerical caveats).
- **No `from x import *`.** Ever.
- **No global state.** Configs are passed explicitly.
- **Logging via the `logging` module**, not `print`, except in CLI summary output.
- **Random seeds are explicit and logged.** Every stochastic operation takes a seed and the seed appears in the run output.
- **Errors are loud.** Don't silently fall through. If yfinance returns empty data, raise. If a feature produces all NaNs, raise. If a model fails to converge in all restarts, raise.
- **Tests live in `tests/`** and run via `uv run pytest`. Inference math tests are P0; the rest grow over time.
- **Lint via `ruff`** with a permissive config. Format on save in VS Code.

---

## 7. Antipatterns (forbidden)

- ❌ Hardcoding state count, e.g., `n_states=3`.
- ❌ Single-fit training (`model.fit(X)` once and trusting it).
- ❌ Calling `yf.download(..., start=..., end=...)` for hourly data.
- ❌ Computing features inside the model layer.
- ❌ Storing fitted scalers separately from the model they belong to.
- ❌ Using future data in walk-forward validation.
- ❌ Treating `["Bearish", "Neutral", "Bullish"]` as the canonical labels.
- ❌ `plt.show()` inside library functions.
- ❌ Catching exceptions silently.
- ❌ Adding a Bitcoin-specific code path. Asset behavior lives in YAML.
- ❌ Adding a new dependency without an entry in the Decision Log.

---

## 8. Phase 1 instructions for Claude Code (start here)

When this repo is empty or near-empty, your job is to scaffold and implement Phase 0 + Phase 1 end-to-end. **Do not jump ahead to Phase 2+ on the first pass.** Robustness and selection come after a clean baseline runs.

**Phase 0 deliverables (Setup):**

1. `pyproject.toml` initialized with `uv`, Python ≥3.11, project metadata, dependencies, dev-dependencies, `[project.scripts]` entry point `rde = "rde.cli:main"`.
2. Directory skeleton matching section 4. Empty `__init__.py` files where needed.
3. `.gitignore` (Python defaults + `.venv`, `results/` except `.gitkeep`, `.idea/`, `.vscode/` partial, `*.parquet` in cache).
4. `README.md` with a one-line description and a "Getting Started" pointing at `Setup Guide` in Notion.
5. `LICENSE` (MIT).

**Phase 1 deliverables (BTC baseline, end-to-end):**

1. `rde.data.yfinance_source.YFinanceSource` honouring `period="730d"`, `interval="1h"` with parquet caching.
2. `rde.features` with `LogReturns`, `RollingVolatility`, `SmoothedReturns`, `FeaturePipeline`.
3. `rde.models.hmm.train_hmm` (single-fit acceptable in Phase 1; multi-restart added in Phase 2).
4. `rde.inference.viterbi.viterbi_decode`.
5. `rde.viz.static_plots.plot_price_with_regimes`.
6. `rde.evaluation.transition.stationary_distribution` and a basic `diagnostics.txt` writer.
7. `rde.labeling.ranking.rank_states`.
8. `rde.config` schema + YAML loader.
9. `rde.cli` that ties it all together.
10. `configs/btc.yaml`.
11. One smoke test: `tests/test_smoke_btc.py` that imports the modules and runs the pipeline on a tiny synthetic DataFrame (does NOT hit yfinance in CI).

**Phase 1 acceptance criteria:**

```bash
uv sync
uv run rde run --config configs/btc.yaml
```

…produces:

- `results/BTC-USD/plots/price_with_regimes.png`
- `results/BTC-USD/diagnostics.txt` (transition matrix, state stats, heuristic labels with disclaimer)
- `results/BTC-USD/regimes.parquet`
- A short stdout summary

…and the smoke test passes:

```bash
uv run pytest tests/test_smoke_btc.py
```

---

## 9. Current state (as of Phase 50)

**Phases 0–36 are complete on `main`, plus the post-Phase-36 audit branch (`audit/post-phase-36-improvements`).** 1297 tests passing. The system now includes a full live paper-trading loop with risk guard protection and a complete Streamlit dashboard.

### Post-Phase-36 audit (branch: `audit/post-phase-36-improvements`)

| Item | Summary | Outcome |
|------|---------|---------|
| 2.1 BIC ceiling | Extended n_states [2..10]; n=8 practical optimum via dwell-time criterion | `results/n_states_audit.md`, `docs/findings/2026-05-04_btc_stale_selection.md` |
| 2.2 Synthetic falsification | Gaussian/Student-t/HSMM recovery tests (`@pytest.mark.slow`) | `tests/integration/test_synthetic_recovery.py` |
| 2.3 Numerics audit | 100k/250k/500k sequence stress tests; all hmmlearn inference log-space stable | `tests/numerics/test_log_space_stability.py`, `docs/numerics_audit.md` |
| 2.4 Reproducibility | Golden parquet fixtures; bit-for-bit Viterbi reproducibility enforced | `tests/regression/test_reproducibility.py`, `tests/regression/fixtures/` |
| 2.5 Notebook check | All 3 notebooks confirmed to use current API; no stale prototypes | No code changes |
| 2.6 Docstring sweep | Zero mypy errors, zero annotation gaps on 11 Phase 31–36 files | No code changes |
| 2.7 Notion update | Roadmap, diary, decision log, root status all updated | Notion updated |

### Completed phases

| Phase | Module | Summary |
|-------|--------|---------|
| 0–1 | Core scaffold | Data, features, HMM, inference, viz, labeling, CLI, configs |
| 2 | `models/hmm.py` | Multi-restart Baum-Welch, AIC/BIC selection |
| 3–5 | Various | Plotly, transition heatmap, evaluation harness, walk-forward |
| 6–8 | `models/` | Student-T HMM, GARCH-HMM, semi-Markov (HSMM) |
| 9–11 | `backtest/`, `evaluation/` | Vectorized backtester, metrics, walk-forward V2 |
| 12–14 | `analysis/` | Cross-asset, portfolio (Kelly/vol-target), regime VAR/IRF |
| 15 | `models/` | Spectral init, ensemble HMM |
| 16 | `analysis/risk_metrics.py` | VaR/CVaR/Sortino/Calmar per regime (hard + soft) |
| 17 | `models/change_point.py` | BOCPD with Normal-Gamma conjugate prior |
| 18 | `analysis/information_geometry.py` | KL, JS, Bhattacharyya, Mahalanobis, entropy rate |
| 19 | `models/kalman.py` | Kalman filter + regime-switching blended filter |
| 20 | `analysis/drawdown_control.py` | Drawdown hysteresis, regime position limits |
| 21 | `models/simulation.py` | Gaussian + GARCH-HMM Monte Carlo, scenario aggregation |
| 22 | `analysis/execution.py` | Square-root impact model, urgency schedule, TWAP/VWAP |
| 23 | `analysis/factor_analysis.py` | Weighted PCA per regime, rolling factor exposure |
| 24 | `analysis/cointegration.py` | OU half-life, weighted OLS spread, regime Z-score |
| 25 | `analysis/portfolio_optimization.py` | MVO (min-var/max-Sharpe), efficient frontier, Black-Litterman |
| 26 | `analysis/signal_filtering.py` | Regime-adaptive EMA, HP filter, RTS Kalman smoother |
| 27 | `analysis/transition_prediction.py` | Logistic next-regime predictor, h-step matrix power, dwell time |
| 28 | `analysis/tail_risk.py` | GPD tail fitting, extrapolated VaR/ES, stress scenarios |
| 29 | `analysis/backtest.py` | Regime-conditional strategy engine, tearsheet metrics |
| 30 | `analysis/correlation.py` | Weighted Pearson/Spearman/Kendall, DCC blending, tail dependence |
| 31 | `analysis/pipeline.py`, `analysis/reporting.py`, `analyse_cmd.py` | `AnalysisPipeline` orchestrating all Phase 22–30 modules; `rde analyse` CLI; JSON + Markdown reports |
| 32 | `app/panels_analysis.py`, `app/streamlit_app.py` | Dashboard: 8 new Plotly panels surfacing Phase 31 analysis output |
| 33 | `app/streamlit_app.py`, `analysis/cross_asset.py`, `cli.py` | Current State panel; date range filter; `rde compare --daily` for mixed-market alignment |
| 34 | `trading/` module, `analysis/regime_concordance.py`, `app/panels_live.py`, `configs/` | Paper portfolio, exchange abstraction (MockExchange + BinanceTestnet), regime-change alerting, Live Feed dashboard panel, QQQ/GLD/SOL configs, cross-asset concordance analysis |
| 35 | `trading/loop.py`, `trade_cmd.py`, `cli.py` | `TradingLoop` (OnlineDecoder → strategy → PaperPortfolio → alerts); `rde trade` CLI with live-polling and `--backtest` replay modes |
| 36 | `trading/risk_guard.py`, `app/panels_trade.py`, `app/panels_concordance.py` | `RiskGuard` drawdown + daily-loss monitor integrated into `TradingLoop`; Trade History dashboard panel (equity curve, drawdown, fills, per-regime P&L); Regime Concordance panel (sync heatmap, rolling concordance, lead-lag chart) |
| 37.1 | `evaluation/purged_cv.py` | Purged k-fold CV + combinatorial purged CV with embargo (de Prado AFML ch. 7); causal OnlineDecoder; FoldResult; parquet output |
| 37.2 | `evaluation/skeptics.py` | Skeptic's kit: random baseline, shuffle test, feature ablation, period robustness, cost sensitivity sweep, slippage stress; `skeptics_report.md` |
| 37.3 | `evaluation/baselines.py` | Proper baselines: B&H, vol-targeted B&H, naive momentum, naive vol-regime, 2-state HMM; `compare_model_to_baselines` |
| 37.4 | `evaluation/feature_importance.py` | Permutation feature importance per CV fold; fold-stability scoring (positive-fold fraction ≥ 0.7); `FeatureImportanceResult` |
| 37.5 | `evaluation/honest_tearsheet.py` | Honest tearsheet: Sharpe distribution, MDD distribution, worst-5% fold, cost break-even, capacity estimate, failure modes, baseline comparison; `honest_tearsheet.md` |
| 37b | `research/strategies/vol_target_overlay.py`, `docs/findings/` | Half-dataset stability diagnostic (inter-half ARI=0.742); vol-target overlay Track B (FAIL: -0.155 Sharpe improvement, 294 trades/year); negative result writeup; **research COMPLETE, tagged `v2.1-final-research`** |
| 41 | `configs/btc_daily.yaml`, `scripts/run_phase37_validation.py`, `evaluation/honest_tearsheet.py` | Daily-frequency probe: BTC daily n=8, 27 purged folds. NO-GO: period robustness ARI=0.368 (need ≥0.40); beats only 1/5 baselines (loses to naive momentum 0.892). PASS on: combo CV 0.459±0.323, cost break-even=∞, turnover 19.5/yr. See `docs/findings/phase41_daily_decision_memo.md` |
| 42 | `analysis/multi_asset_allocation.py` | Walk-forward regime-conditional multi-asset MVO: per-asset HMM → posterior-weighted expected returns → joint-cov MVO → optional vol-target overlay; monthly rebalance; `equal_weight_baseline`, `global_min_var_baseline`, `compare_allocations`; 29 tests |
| 42b | `scripts/run_multi_asset_backtest.py`, `docs/findings/phase42_multi_asset_decision_memo.md` | Empirical run BTC/ETH/SPY/GLD daily 2017–2026 (2121 bars). NO-GO: regime_mvo Sharpe=0.450 vs global_min_var=0.937. Regime conditioning adds +0.11 over equal_weight but loses to pure variance minimisation by −0.487 Sharpe. Root cause: mixed crypto+equity portfolio — structural vol difference (BTC 80% vs SPY 15%) dominates any directional tilt from HMM. See `docs/findings/phase42_multi_asset_decision_memo.md` |
| 42c | `analysis/multi_asset_allocation.py` (`regime_informed_min_var`), `docs/findings/phase42_multi_asset_decision_memo.md` | Regime-informed min-var probe: exclude assets with posterior-weighted E[r] < 0 from eligible min-var set. NO-GO: Sharpe=0.410 vs global_min_var=0.937. MDD=67.4% (same as equal_weight) — regime exclusions do not protect against the large crypto drawdowns that define the gap. Root cause: onset latency (HMM is backward-looking at crash onset), min_eligible fallback retains crypto during bear markets. The entire BTC/ETH/SPY/GLD universe is exhausted. See `docs/findings/phase42_multi_asset_decision_memo.md` |
| 43 | `configs/spy_daily.yaml`, `docs/findings/phase43_equity_only_decision_memo.md` | Equity-only probe SPY/GLD/TLT/IEF daily 2004–2026 (5381 bars). PARTIAL POSITIVE: regime_mvo achieves MDD=16.0% vs global_min_var MDD=21.7% — first result where regime conditioning beats pure min-var on drawdown. Calmar: regime_mvo 0.308 vs global_min_var 0.306 (tied). Sharpe still FAIL (0.661 vs 0.929) — strategy too conservative, undershoots rallies. Root cause: strategy design problem (binary on/off), not signal quality problem. Next: Phase 37 purged CV on SPY daily to validate signal stability. See `docs/findings/phase43_equity_only_decision_memo.md` |
| 43b | `docs/findings/phase43b_spy_phase37_validation.md` | SPY daily n=8 Phase 37 NO-GO: ARI=0.311, 0/5 baselines beaten, two_state_hmm dominates by −0.630 margin. n=8 overfit for SPY — BIC audit shows no elbow. |
| 43c | `docs/findings/phase43c_spy_n_sweep.md` | n-states sweep n=2/3/8 on SPY daily. n=3 is best: ARI=**0.393** (just 0.007 below threshold), shuffle margin=−0.027 (nearly neutral), Sharpe=0.587. Single-asset directional still NO-GO but n=3 is best signal quality yet. Phase 44 re-runs equity portfolio with n=3. |
| 44 | `scripts/run_multi_asset_backtest.py`, `results/multi_asset_equity_n3/` | SPY/GLD/TLT/IEF n=3 equity portfolio. regime_informed_min_var Sharpe=0.860 vs global_min_var=0.929. Return higher (7.29% vs 6.64%) but vol also higher (8.48% vs 7.15%) — gap is purely vol, not return. |
| 44b | `scripts/run_multi_asset_backtest.py`, `results/multi_asset_equity_n3_mw60/` | Phase 44b: max_weight=0.60 for rimv (was 1.0). regime_informed_min_var Sharpe=0.837 — slightly worse than Phase 44. Capping concentration after exclusion forces sub-optimal diversification. regime_informed_min_var and global_min_var overlap too much: both prefer TLT/IEF. Regime exclusion approach exhausted. |
| 45 | `analysis/multi_asset_allocation.py` (`regime_tilted_min_var`), `docs/findings/phase45_rtmv_decision_memo.md` | **FIRST REGIME STRATEGY TO BEAT GLOBAL MIN-VAR.** Convex combination: w=(1-λ)×w_minvar + λ×w_regime. rtmv_l05 Sharpe=0.934 > global_min_var=0.929. MDD monotonically improves with λ (21.7%→20.4%). Calmar monotonically improves (0.306→0.362). Conditional PASS pending OOS validation (Phase 45b). |
| 45b | `scripts/run_multi_asset_backtest.py` (--eval-start-date), `results/multi_asset_equity_n3_oos/` | **GO.** OOS (2016–2026): ALL RTMV variants beat global_min_var (rtmv_l30 Sharpe=0.947 > 0.882). Lambda ordering shifts OOS (l30 best vs l05 in full period) but any lambda beats no tilt. MDD and Calmar improvements hold OOS. See `docs/findings/phase45_rtmv_decision_memo.md`. |
| 46 | `evaluation/vol_forecasting.py`, `scripts/run_vol_forecast_test.py`, `tests/edge_validation/test_vol_forecasting.py` | Vol forecasting quality test: HMM posterior-weighted vol vs EWMA vs HistVol at h=5/10/21 bars. 16 tests. Script: `run_vol_forecast_test.py --config`. |
| 47 | `scripts/run_rtmv_cv_validation.py`, `docs/findings/phase47_rtmv_cv_decision_memo.md` | **CONDITIONAL GO** (λ=0.05): fold-consistency PASS (3/5=60%), cost break-even PASS (80.6 bps >> 20 bps min), shuffle FAIL (p=0.130, margin +0.012 — real beats shuffle mean but 13% of shuffles beat real). λ=0.30 fails all 3 tests (wrong period). Signal is real but small (+0.005 Sharpe over GMV). See `docs/findings/phase47_rtmv_cv_decision_memo.md`. |
| 48 | `analysis/multi_asset_allocation.py` (`compute_rtmv_weights_now`), `trading/multi_asset_portfolio.py`, `trading/rtmv_rebalancer.py`, `scripts/run_rtmv_live.py` | Live deployment skeleton: `compute_rtmv_weights_now` (single-step live weight computation from a lookback window), `MultiAssetPortfolio` (N-position paper portfolio with weight-based rebalance + fills), `RTMVRebalancer` (daily-polling monthly-rebalancing loop with drawdown halt, backtest mode, live polling mode), `run_rtmv_live.py` (backtest: `--mode backtest`, live: `--mode live`). 25 new tests. Backtest result: Sharpe=0.875, MDD=−21.5%, Ann Return=6.4%, N Rebalances=212. |
| 49 | `app/panels_rtmv.py`, `app/streamlit_app.py` | RTMV Portfolio Monitoring dashboard panel: equity+drawdown subplot, stacked area weight trajectory, recent fills table (last 50), 6 summary metrics (equity, Sharpe, MDD, ann return, N rebalances, status). Reads `results/rtmv_live/snapshots.parquet` and `fills.parquet`. 14 new tests in `tests/test_panels_rtmv.py`. |
| 50 | `trading/rtmv_rebalancer.py`, `evaluation/baselines.py`, `docs/findings/phase50*.md` | Three-quant parameter study. **50a** adaptive λ NO-GO (−0.003 Sharpe; n=3 posteriors too concentrated). **50b** KL-triggered rebalance NO-GO (fires on 87% of bars; HMM refit window drift dominates KL signal). **50c** two GOs: halt=20% raised to 25% (+0.027 Sharpe, natural MDD is 21.5% so 20% was premature); RTMV beats risk parity baseline (+0.021 Sharpe, +0.035 Calmar, −1.1pp MDD). 1535 tests. |

### Full pipeline (paper trading)

```bash
# 1. Generate model + regimes
uv run rde run --config configs/btc.yaml --save-model results/BTC-USD/model.pkl

# 2. Run analysis reports
uv run rde analyse --config configs/btc.yaml

# 3. Cross-asset comparison
uv run rde compare --result results/BTC-USD --result results/ETH-USD --result results/SPY --daily

# 4. Paper-trade backtest on historical data
uv run rde trade --model results/BTC-USD/model.pkl --config configs/btc.yaml --backtest

# 5. Live paper-trading loop (polls yfinance every 60s)
uv run rde trade --model results/BTC-USD/model.pkl --config configs/btc.yaml

# 6. Dashboard
uv run streamlit run app/streamlit_app.py
```

### Phase 37 validation pipeline

```bash
# Run full Phase 37.1-37.5 validation on BTC
uv run python scripts/run_phase37_validation.py --config configs/btc.yaml \
    --n-states 8 --train-bars 4000 --test-bars 500 --n-restarts 3

# Outputs:
#   results/BTC-USD/purged_cv_{date}.parquet
#   results/BTC-USD/combinatorial_cv_{date}.parquet
#   results/BTC-USD/skeptics_report.md       ← adversarial tests pass/fail
#   results/BTC-USD/honest_tearsheet.md      ← STOP and read this before Phase 38
```

### Phase 37b outcome (2026-05-05)

**RESEARCH COMPLETE — `v2.1-final-research`**

Phase 37b ran the stability diagnostic and attempted a vol-target overlay strategy (Track B). Both the directional signal and the overlay fail the 5 bps cost break-even required for Trading212 deployment. The research has been archived with a full negative result writeup.

Key findings:
- **Regime structure IS stable** (inter-half ARI = 0.742) — non-stationarity is NOT the cause of failure.
- **Track B overlay fails**: Sharpe improvement = -0.155, 294 trades/year (need < 20). Root cause: 8-state posterior label permutation within the 24-bar averaging window causes continuous exposure oscillation despite instantaneous confidence of 0.945.
- **The edge is real but not deployable**: shuffle/random-baseline margins > 1.6 confirm real temporal structure; cost threshold is the binding constraint.
- Full writeup: `docs/findings/negative_result_writeup.md`, `docs/findings/track_b_decision_memo.md`

### Phase 41 outcome (2026-05-06)

**Daily probe NO-GO.** BTC-USD daily bars, n=8 (BIC-selected), 4,229 bars (2014–2026). Combinatorial CV Sharpe=0.459±0.323 (PASS), turnover=19.5/yr (PASS), cost break-even=∞ (PASS) — but period robustness ARI=0.368 (need ≥0.40, FAIL) and the model beats only 1 of 5 baselines (naive momentum achieves Sharpe=0.892 vs model's 0.437). The binding failure: directional momentum dominates daily BTC, and the 8-state HMM does not capture it. See `docs/findings/phase41_daily_decision_memo.md`.

### Multi-asset allocation outcome (2026-05-08)

**NO-GO (all variants).** BTC/ETH/SPY/GLD daily 2017–2026:
- regime_mvo (no vol-target) Sharpe=0.450 vs global_min_var=0.937 (Phase 42b)
- regime_informed_min_var Sharpe=0.410 vs global_min_var=0.937 (Phase 42c)
- MDD for all regime strategies = 67.4% vs global_min_var = 31.9%

Root cause: structural vol disparity (BTC ~80%, SPY ~15%) — min-var structurally
allocates ~70–80% to low-vol assets, making it impossible for regime conditioning
to bridge the MDD gap. Regime exclusions do not protect drawdowns because the
HMM is backward-looking at crash onset. The entire BTC/ETH/SPY/GLD universe is
exhausted. See `docs/findings/phase42_multi_asset_decision_memo.md`.

### Picking up from here

**BTC/ETH directions exhausted.** Equity-only (Phase 43) shows a partial
positive: regime_mvo MDD=16.0% beats global_min_var MDD=21.7% — first
drawdown improvement from regime conditioning. Calmar tied (0.308 vs 0.306).
Sharpe fails (0.661 vs 0.929) due to strategy over-conservatism, not signal
weakness. **Phase 37 purged CV validation on SPY daily is running.**

**Phase 47 CONDITIONAL GO — RTMV(λ=0.05) passes fold-consistency and cost tests:**
- Fold-consistency: 3/5 folds RTMV wins (60%, criterion ≥60%) — PASS
- Cost break-even: 80.6 bps (criterion ≥20 bps) — **PASS** — 8× safety margin over ~10 bps practical cost
- Shuffle robustness: p=0.130 (criterion <0.10, margin +0.012) — FAIL (marginal; n=100 shuffles may be insufficient)
- λ=0.30 is NOT the right lambda for full-period — it fails all 3 tests (selected on 2016–2026 only)

**Phase 50 COMPLETE — parameter study complete.** 1535 tests passing.

Phase 50 key results:
- Halt threshold raised 20% → 25%: Sharpe improves 0.876 → **0.903** (natural MDD is 21.5%; 20% was premature)
- RTMV beats risk parity: Sharpe +0.021, Calmar +0.035, MDD −1.1pp
- Adaptive λ and KL-triggered rebalance both NO-GO

Run the updated backtest and launch dashboard:
```bash
make backtest-rtmv          # SPY/GLD/TLT/IEF backtest → results/rtmv_live/
make dashboard              # Streamlit dashboard with RTMV Portfolio panel
```
Switch to live polling:
```bash
make live-rtmv              # polls yfinance every 3600s, rebalances monthly (halt=25%)
```

**Confirmed next direction:**
→ Run `make live-rtmv` to begin live paper trading. Monitor for 1–3 months. Target live Sharpe ≈ 0.903. Drawdown halt now at 25% — will only trigger in tail scenarios (2008-style). Next research directions: (1) online-posterior KL monitor (decouple trigger from refit); (2) regime-conditional λ keyed on dominant state identity rather than entropy; (3) universe expansion beyond SPY/GLD/TLT/IEF.

The Notion roadmap and tasks database are the source of truth for sequencing. Update Notion as work progresses.

---

## 10. Working with Claude Code

- **Read this file first.** It is the project's invariants.
- **When making architectural decisions, propose them in the chat, then add an entry to the Decision Log in Notion before implementing.** A code change that contradicts CLAUDE.md without an updated CLAUDE.md is a bug.
- **When proposing new features that go beyond v1.0**, flag them as post-v1.0 and add them to the Notion parking lot rather than implementing.
- **When uncertain, say so explicitly.** Do not guess at numerical thresholds, paper conventions, or library behavior. Verify or ask.
- **When fixing existing notebooks** (the project knowledge contains a working but limited reference notebook), treat them as informational, not prescriptive. The new system supersedes them.
