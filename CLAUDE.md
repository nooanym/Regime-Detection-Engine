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

## 9. Current state (as of Phase 32)

**Phases 0–32 are complete on `main`.** 1108 tests passing. The original v1.0 roadmap (Phases 0–5) is long done; the project has grown into a deep analytics library with a full interactive dashboard.

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
| 32 | `app/panels_analysis.py`, `app/streamlit_app.py` | Dashboard: 8 new Plotly panels surfacing Phase 31 analysis output (tail risk, factors, transitions, correlation, portfolio, cointegration, execution) |

### Running the dashboard

```bash
uv run rde run --config configs/btc.yaml          # generate regimes + signals
uv run rde analyse --config configs/btc.yaml      # generate analysis_report.json
streamlit run app/streamlit_app.py                # open dashboard
```

### Picking up from here

The analytics layer and dashboard are complete. Possible next directions (coordinate with Notion):

- **Phase 33+**: Live data feed — replace `YFinanceSource` with Binance/Coinbase WebSocket streaming
- **Phase 33+**: Online regime inference — `OnlineDecoder` consuming a real-time bar queue
- **Phase 33+**: Paper trading loop — regime → signal → ccxt order on Binance testnet
- **Phase 33+**: Additional asset configs (ETH, SPY, QQQ) and cross-asset regime concordance
- **v2.0 release tag** — analytics layer + dashboard complete; add git tag once integration test passes

The Notion roadmap and tasks database are the source of truth for sequencing. Update Notion as work progresses.

---

## 10. Working with Claude Code

- **Read this file first.** It is the project's invariants.
- **When making architectural decisions, propose them in the chat, then add an entry to the Decision Log in Notion before implementing.** A code change that contradicts CLAUDE.md without an updated CLAUDE.md is a bug.
- **When proposing new features that go beyond v1.0**, flag them as post-v1.0 and add them to the Notion parking lot rather than implementing.
- **When uncertain, say so explicitly.** Do not guess at numerical thresholds, paper conventions, or library behavior. Verify or ask.
- **When fixing existing notebooks** (the project knowledge contains a working but limited reference notebook), treat them as informational, not prescriptive. The new system supersedes them.
