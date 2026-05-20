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

## 9. Current state (as of Phase 81)

**Phases 0–36 are complete on `main`, plus the post-Phase-36 audit branch (`audit/post-phase-36-improvements`).** 1577 tests passing. The system now includes a full live paper-trading loop with risk guard protection and a complete Streamlit dashboard.

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
| 51 | `analysis/multi_asset_allocation.py`, `trading/rtmv_rebalancer.py`, `scripts/compare_lambda_strategies.py` | Regime-conditional λ via per-asset rank averaging. NO-GO: best +0.003 Sharpe (threshold +0.010). Root cause: SPY rank-0 (bear) + TLT rank-2 (bond-bull) cancel → mean rank stays neutral. See `docs/findings/phase51_regime_conditional_lambda.md`. |
| 51b | `analysis/multi_asset_allocation.py` (`lambda_proxy_asset`), `trading/rtmv_rebalancer.py`, `scripts/compare_lambda_strategies.py` | SPY-proxy λ — use only SPY's dominant state rank to set portfolio λ. `spy_rank_bull=[0.02,0.05,0.10]` **GO: +0.011 Sharpe** (Sharpe=0.8948 vs baseline 0.8838, MDD −0.2pp). 1548 tests. See `docs/findings/phase51b_spy_proxy_lambda.md`. |
| 52a | `scripts/run_phase52_universe_expansion.py`, `results/phase52/` | Bond maturity ladder: 4/5/6-asset universe expansion. **GO: 5-asset (SPY/GLD/SHY/IEF/TLT) wins** — spy_rank_bull Sharpe=0.9726 (+0.0778 vs 4-asset baseline). MDD −15.6% vs −21.3% (−5.7 pp). Adding SHY spans the full duration curve. 6-asset (+TIP) improves MDD slightly (−14.9%) but reduces Sharpe (0.9665). See `docs/findings/phase52a_bond_ladder.md`. |
| 53 | `trading/rtmv_rebalancer.py`, `scripts/run_phase53_kl_monitor.py`, `results/phase53/` | Online-posterior KL monitor. **NO-GO.** All thresholds [0.05–2.0] fire 45.7×/yr — `kl_min_bars_between_triggers=5` is binding, not KL. Forward filter on n=3 daily HMM oscillates every ~5 bars regardless. Confidence gate (`kl_min_dominant_confidence=0.50`) did not reduce trigger rate. See `docs/findings/phase53_kl_monitor.md`. |
| 54 | `models/joint_hmm.py` (in progress) | Joint 5D HMM on SPY/GLD/SHY/IEF/TLT return vector simultaneously. Hypothesis: single portfolio-level HMM produces more stable posteriors than 5 independent per-asset HMMs, addressing Phase 53's forward-filter oscillation. GO threshold: Sharpe ≥ 0.920 using joint-state posterior as RTMV signal. **IN PROGRESS** — results pending. |
| alpha-hunt | `trading/carry_executor.py`, `scripts/funding_carry_backtest.py`, `scripts/monitor_carry.py`, `docs/findings/alpha_hunt_synthesis.md` | Alpha hunt parallel research track. **TIER 1 CONFIRMED: crypto funding carry.** ETH carry Sharpe ~16, Ann Return ~14.7%, MDD ~−0.5%, 85.6% positive 8-hour periods, worst year 2022 still positive (+0.8%). BTC carry Sharpe ~12, Ann Return ~12.1%. CarryPortfolio + CarryStrategy classes operational. Live monitor running. Carry is structurally uncorrelated with RTMV and can run alongside it. Combined portfolio (70% RTMV + 15% ETH carry + 10% BTC carry + 5% reserve) expected combined Sharpe ~1.5–2.0, MDD < 12%. GBTC arb and LETF decay overlay reports pending. See `docs/findings/alpha_hunt_synthesis.md`. |
| 54 | `src/rde/analysis/multi_asset_allocation.py` (`compute_rtmv_weights_now_joint`), `src/rde/trading/rtmv_rebalancer.py`, `tests/test_joint_hmm.py` | Joint 5D HMM on SPY/GLD/SHY/IEF/TLT simultaneously. **CONDITIONAL NO-GO.** Sharpe 0.952 clears GO threshold (0.920) but does NOT beat Phase 52a independent-HMM baseline (0.973). Gap of −0.021 explained by information dilution in 5D joint space and `diag` covariance discarding cross-asset co-movements. Phase 52a (spy_rank_bull, 5-asset, independent HMMs) remains live deployment target. Makefile: `make backtest-rtmv-joint`. See `docs/findings/phase54_joint_hmm.md`. |
| 55 | `scripts/run_phase55_regime_carry.py`, `src/rde/trading/carry_executor.py`, `scripts/run_carry_live.py` | Regime-scaled ETH carry (SPY HMM rank → position scale). **CONDITIONAL GO — bull-only (1×/1×/1.5×) variant.** Full regime-scaled (0.5×/1×/1.5×): NO-GO (Sharpe 16.205 < 16.5 threshold). Bull-only: Sharpe 16.395, Ann 18.49%, Cum +196.2% (+53.9pp vs flat). The 0.5× bear reduction hurts — ETH carry stays positive even in SPY bear regimes. `run_carry_live.py` now accepts `--regime-scale bull-only` (default). `CarryStrategy.step()` and `run_live()` wired to pass SPY rank. See `docs/findings/phase55_regime_carry.md`. |
| 56 | `scripts/vrp_analysis.py`, `docs/findings/phase56_vrp.md` | Vol risk premium (VRP) harvesting via SVXY. Regime-only filter (invest when SPY rank=2): Sharpe **0.926**, avoids 74% of blow-up days (including Feb 2018 and March 2020 COVID). Full dual filter (VRP + regime): NO-GO (Sharpe 0.315 — VRP threshold is redundant since rank-2 days are 98.2% positive VRP already). Main miss: SVXY structural break in Feb 2018 (−1× → −0.5× VIX futures). **PARTIAL GO** — Phase 56b investigates post-2018 period + VXX-short. See `docs/findings/phase56_vrp.md`. |
| 58 | `scripts/run_multistrat.py`, `docs/findings/phase58_multistrat.md` | Multi-strategy executor combining ETH/BTC funding carry + LETF decay short. **GO.** Cross-strategy correlation = **−0.009** (structurally independent alpha sources). Optimal 80/20 carry/LETF allocation: Sharpe 7.611, MDD −0.95%, Ann 8.27%. Combined MDD substantially better than LETF leg alone (−3.36%) while carry Sharpe dilution is modest. See `docs/findings/phase58_multistrat.md`. |
| 59 | `docs/findings/phase59_expanded_carry.md` | Expanded crypto carry universe: SOL, AVAX, BNB, DOGE added to BTC+ETH basket. **NO-GO.** Best expansion (BTC+ETH+DOGE) Sharpe 15.82 < 16.0 threshold. Root cause: dilution arithmetic — any alt with individual Sharpe 8-12 pulls the combined Sharpe below the BTC+ETH baseline of 15.86. BNB permanently excluded (−0.8% mean carry, 22% positive periods). DOGE is the best candidate (carry 13.1%, corr_ETH 0.598) — monitor for 6+ months sustained carry before revisiting. Recommended next: dynamic carry-weighting within BTC+ETH (Phase 60). |
| 56b | `scripts/run_phase56b_svxy_post2018.py`, `docs/findings/phase56b_svxy_post2018.md` | Post-2018 SVXY + VXX-short regime filter (Phase 56 follow-up). **NO-GO.** Post-2018 SVXY (−0.5× structure): regime filter Sharpe 0.651, MDD −20% — SVXY's halved leverage eliminates the carry differential. VXX-short: Sharpe 0.845 at 0% borrow, 0.829 at 1% — ceiling is below the 1.200 threshold at any borrow cost. Signal is real (+0.61 Sharpe vs B&H on SVXY, +0.48 vs B&H on VXX-short) but insufficient standalone. Recommended path: combine VXX-short as a 5–10% allocation in the full multi-alpha portfolio (Phase 61) where the regime filter's MDD reduction provides diversification value. |
| 60 | `scripts/run_phase60_carry_weight.py`, `docs/findings/phase60_carry_weight.md` | Dynamic carry-weighted BTC+ETH allocation. **STRONG GO.** carry_weighted Sharpe **18.129** vs equal_weight 17.499 (+0.630). MDD improves −0.85% → −0.46% (exits 5.8% of periods when both carries turn non-positive). ETH structurally higher-carry: avg split 51.2% ETH / 48.8% BTC; bull years 65% ETH, bear years 30% ETH. max_carry (100% to winner) Sharpe 17.725 — worse than proportional weighting (higher vol offsets higher return). Integrated: `CarryStrategy` now defaults to `carry_weighted=True`, `run_carry_live.py` accepts `--no-carry-weighted` to disable. |
| 57 | `src/rde/analysis/multi_asset_allocation.py` (`momentum_tilt_scale`), `src/rde/trading/rtmv_rebalancer.py`, `scripts/run_phase57_momentum.py` | Cross-asset (12m−1m) momentum overlay on RTMV weights. **GO.** `momentum_tilt_scale=0.03`: Sharpe **1.0008** (+0.028 vs Phase 52a 0.9726, 2.7× +0.010 threshold). MDD unchanged −15.5%. Signal adds z-score momentum tilt additively to `w_combined`; clips to [0,∞) then renormalises. Deployed to `make live-rtmv` and `make backtest-rtmv-5asset` (`--momentum-tilt-scale 0.03`). 1579 tests passing. See `docs/findings/phase57_momentum.md`. |
| 64 | `scripts/run_phase64_letf_filters.py`, `docs/findings/phase64_letf_filters.md` | LETF decay with SPY regime + QQQ momentum filter. **NO-GO** — always-in confirmed optimal. Filters reduce Sharpe: best filter 4.188 vs always-in 4.724. **Critical finding: delta-neutral pair (SHORT TQQQ + LONG 3×QQQ) Sharpe 4.724 vs pure TQQQ short 1.137 (4× improvement)**. Phase 65 formalizes. |
| 65 | `scripts/run_phase65_letf_pair.py`, `docs/findings/phase65_letf_pair.md` | Delta-neutral LETF pair strategy (SHORT TQQQ + LONG 3×QQQ). **STRONG GO.** Full 2010–2026 backtest: Sharpe **4.887**, Ann 17.61%, Vol 3.60%, MDD −1.36%, Calmar 12.967, Cum **+1206%**. **Every calendar year was positive.** 2022 bear market returned +44.27% (vol spike = more decay premium). 107% of P&L comes from theoretical 4.5σ²_QQQ vol-drag term. `make letf-pair`. |
| 67 | `scripts/run_letf_pair_live.py` | LETF pair paper-trading executor. Three modes: `signal` (today's P&L + YTD), `backtest` (full tearsheet), `monitor` (daily parquet log). Today: Pair +0.11%, YTD +7.05%, Max DD −0.09%. `make letf-pair-signal` / `make letf-pair-monitor`. |
| 63 | `scripts/run_rtmv_cv_validation.py` (`run_momentum_cv_validation`), `results/rtmv_cv_validation/phase63_momentum_cv_20260518.md` | **STRONG GO.** Phase 57 momentum_tilt_scale=0.03 passes all 3 CV tests on 5-asset SPY/GLD/SHY/IEF/TLT (2004–2026): fold-consistency 3/5 PASS, cost break-even ∞ PASS (advantage holds at all costs including 200 bps), shuffle p=**0.0000** PASS (0/50 shuffled z-score permutations beat real — strongest possible result). Momentum Sharpe 0.9713 vs baseline 0.9438 (+0.027). Shuffle mean 0.3401 (margin +0.631) confirms cross-sectional z-score ordering is the signal, not HMM seed luck. `momentum_tilt_scale=0.03` confirmed as live baseline. |
| 68 | `scripts/run_phase68_multistrat_live.py`, `results/phase68_multistrat/` | Unified multi-strategy executor: 80% carry + 15% LETF pair + 5% RTMV. **STRONG GO.** Combined Sharpe **11.0483** (threshold 7.0), Ann 14.79%, Vol 1.34%, MDD **−0.21%**, Calmar 69.35. **Every calendar year positive** (2020–2026 YTD). 2022 stress: carry +2.1%, LETF +44.3%, RTMV −12.2% → combined **+6.8%**. Diversification uplift: combined Sharpe 11.05 beats best individual (carry 8.96) by +2.09. Cross-strategy correlations: carry↔LETF=−0.077, carry↔RTMV=−0.001, LETF↔RTMV=−0.019 (three structurally independent alpha sources). `make multistrat-backtest` / `make multistrat-signal`. |
| 69 | `scripts/run_phase69_adaptive_threshold.py`, `docs/findings/phase69_adaptive_carry_threshold.md` | Regime-adaptive carry entry threshold. **NULL RESULT (technically GO, practically inert).** Raising bear-regime entry from 5% to 6–10% changes only 3–4 entry decisions over 6,989 eight-hour periods. Root cause: the 90-period rolling carry window already smooths out low-carry spikes — the rolling mean stays above 5% even in bear regimes because carry momentum is persistent. The carry MDD of −0.41% is a structural floor for this architecture. Phase 62 config (flat 5% entry, bull-only 1.5× scale, carry_weighted=True) remains optimal. No changes recommended. |
| 70 | `scripts/run_phase70_letf_universe.py`, `docs/findings/phase70_letf_universe.md` | LETF pair universe expansion. **PARTIAL GO.** Individual pairs: TQQQ/QQQ Sharpe 4.58, SOXL/SOXX 4.33, UPRO/SPY 3.25 — none clears 5.0. **Equal-weight 3-pair combo: Sharpe 5.47, Ann 21.05%, Vol 3.85%, MDD −1.74%, Calmar 12.08.** Every year positive. SOXL↔UPRO correlation=0.14 (near-zero: semiconductor vol is idiosyncratic vs broad equity). Gross vol-drag scales as σ²: SOXX 49%/yr > QQQ 33%/yr > SPY 16%/yr. Higher costs on SOXL (3.40%/yr borrow+expense) prevent single-pair GO. **Recommendation (Phase 71): replace single TQQQ/QQQ leg in Phase 68 with 3-pair combo.** `make letf-universe`. |
| 66 | `scripts/run_phase66_portfolio_frontier.py`, `docs/findings/phase66_portfolio_frontier.md` | RTMV↔LETF pair correlation + portfolio frontier. **Key findings:** LETF↔RTMV correlation = +0.027 (near-zero, NOT negative in bear regimes — bear: +0.017, neutral: +0.082, bull: +0.061). The 2022 hedge hypothesis is NOT confirmed at daily frequency. Carry↔LETF = −0.084 (slight negative). **Best allocation: 80% carry + 15% LETF + 5% RTMV** (Sharpe 9.828, MDD −0.55%). In 2022 stress test: carry +0.99%, LETF +44.27%, RTMV −14.56% — all allocations positive even with RTMV at 25%. `make portfolio-frontier`. |
| 62 | `scripts/run_phase62_regime_carry_weight.py`, `docs/findings/phase62_regime_carry_weight.md` | Regime-scaled carry-weighted BTC+ETH (Phase 60 + Phase 55 combined). **GO.** carry_weighted_bull_only (1×/1×/1.5× SPY rank): Ann **17.55%** (+3.42% over carry_weighted_flat), Cum **180.7%** (+48.3pp), MDD −0.46%. Sharpe 17.943 (slightly below flat 18.129 — scaling up in bull markets raises vol). Full regime (0.5×/1×/1.5×): confirmed again that bear reduction hurts (−0.31% ann, slightly lower MDD −0.33%). **Definitive carry config deployed:** `CarryStrategy(carry_weighted=True, regime_scale={0:1.0, 1:1.0, 2:1.5})` — already the default. |
| 70 | `scripts/run_phase70_letf_universe.py`, `docs/findings/phase70_letf_universe.md` | LETF pair universe expansion. **PARTIAL GO.** Individual pairs: TQQQ/QQQ Sharpe 4.58, SOXL/SOXX 4.33, UPRO/SPY 3.25 — none clears 5.0. **Equal-weight 3-pair combo: Sharpe 5.47, Ann 21.05%, Vol 3.85%, MDD −1.74%, Calmar 12.08.** Every year positive. SOXL↔UPRO correlation=0.14 (near-zero: semiconductor vol is idiosyncratic vs broad equity). Gross vol-drag scales as σ²: SOXX 49%/yr > QQQ 33%/yr > SPY 16%/yr. Higher costs on SOXL (3.40%/yr borrow+expense) prevent single-pair GO. **Recommendation (Phase 71): replace single TQQQ/QQQ leg in Phase 68 with 3-pair combo.** `make letf-universe`. |
| 71 | `scripts/run_phase68_multistrat_live.py` (upgraded), `docs/findings/phase71_letf_3pair_upgrade.md` | 3-pair LETF upgrade to multi-strategy portfolio. **STRONG GO.** Replace single TQQQ/QQQ LETF leg with equal-weight TQQQ/QQQ + SOXL/SOXX + UPRO/SPY. Combined portfolio Sharpe **11.5636** vs Phase 68 baseline 11.0483 (+0.515). Ann Return 15.76% (+0.97pp), MDD −0.20% (improved). LETF combo leg Sharpe 7.06 (vs single pair 6.24). Every calendar year positive, 2022 stress: combined +7.30%. All cross-strategy correlations remain near-zero (carry↔letf −0.097, carry↔rtmv −0.001, letf↔rtmv −0.012). `make multistrat-backtest`. |
| 72 | `scripts/run_phase72_letf_regime_filter.py`, `docs/findings/phase72_letf_regime_filter.md` | SPY HMM regime filter on 3-pair LETF combo. **NO-GO (hypothesis confirmed).** Bull-only 15× (1×/1×/1.5×) Sharpe=5.7997 vs flat=5.4672 — does not clear 7.06 threshold. Root cause: LETF decay earns from VOLATILITY, not direction. Bear regimes (rank=0) produce ann LETF return=**61.02%**, neutral=20.26%, bull=8.91%. SPY rank=2 (bull/low-vol) is the worst regime for vol-drag premium. Scaling UP in bull (when premium is lowest) cannot improve Sharpe meaningfully. Bear-reduce variant (0.5×/1×/1.5×) achieves highest Sharpe improvement (+0.96) by reducing exposure when LETF earns MOST — confirming the inverse relationship. Always-in flat remains optimal. `make letf-regime-filter`. |
| 73 | `scripts/run_phase73_sol_carry.py`, `docs/findings/phase73_sol_carry.md` | SOL perpetual funding carry re-test (2024-2026 data). **CONDITIONAL GO (carry-basket NO-GO, phase threshold fails).** SOL 2024-present mean ann carry = 5.3% (improved from 0.9% full-period), worst year still 2022 (-30.1%). Best SOL-expanded raw basket (btc_eth_sol carry-wt) Sharpe 16.831 — clears 16.0 raw threshold but trails btc_eth baseline (17.430). Best regime-scaled SOL basket Sharpe 17.601 — does not beat Phase 62 live baseline 17.943 (delta -0.342). Critical new finding: SOL/ETH correlation jumped from 0.258 (2021+) to **0.695 (2024+)**, eliminating SOL's diversification advantage vs Phase 59. BTC+ETH basket remains optimal. `make carry-phase73`. |
| 74 | `scripts/run_phase74_portfolio_reopt.py`, `docs/findings/phase74_portfolio_reopt.md` | Portfolio frontier re-optimisation with upgraded 3-pair LETF leg. **NO-GO — 80/15/5 remains optimal.** Fine grid search (63 combinations tested) over carry=[50%..90%], letf=[5%..35%], rtmv=[2%..15%]. Best unconstrained: **75/20/5 Sharpe 11.5794** vs baseline 11.5636 — improvement +0.0158, need +0.09 to GO. Root cause: carry (Sharpe 8.96) dominates combined Sharpe; LETF's higher vol (5.20%/yr) dilutes portfolio Sharpe when its weight rises above 15%. Carry↔LETF correlation −0.097 provides diversification but not enough to overcome vol dilution at higher LETF weights. 2022 stress improves with more LETF (9.35% at 75/20/5 vs 7.30% at 80/15/5). Phase 71 baseline allocation **80% carry / 15% LETF / 5% RTMV** confirmed as optimal. `make portfolio-reopt`. |
| 75 | `scripts/run_phase75_intraday_carry.py`, `docs/findings/phase75_intraday_carry.md` | Intraday UTC window filtering + lag-1 carry momentum filter. **GO (lag-1 momentum filter).** Window skipping NULL RESULT: skipping any fixed UTC window (00:00/08:00/16:00) costs 4.1–4.3 Sharpe — 33% fewer compounding periods overwhelms the tiny quality gain (window carry gap only 0.9pp: 12.7% vs 13.6%). **Lag-1 momentum filter GO:** carry exhibits very strong autocorrelation (lag-1=0.793, lag-3=0.675). Skipping a period when the previous combined rate ≤ 0 achieves Sharpe **18.5728** (+0.4434 vs 18.1294 baseline), Ann Return 14.19% (+0.06pp), MDD **−0.22%** (was −0.46%). The filter removes ~16% of periods where carry turns negative without sacrificing return. `make carry-phase75`. See `docs/findings/phase75_intraday_carry.md`. |
| 76 | `scripts/run_phase76_carry_spread.py`, `src/rde/trading/carry_executor.py`, `docs/findings/phase76_carry_spread.md` | ETH-BTC carry spread signal + Phase 75 deployment. **Part A (filter deployment): DONE.** Phase 75 momentum filter deployed to `CarryStrategy(momentum_filter=True)`. **Part B: STRONG GO.** spot_spread_dynamic (ETH weight = clip(eth_rate/(btc_rate+ + eth_rate+), 0.40, 0.80)) Sharpe **19.7283**, MDD **−0.02%**. ETH is above 1.5× BTC carry for 21.2% of periods — instantaneous proportional overweighting of ETH captures a structural premium that rolling 90-period weights smooth away. `make carry-phase76`. |
| 77 | `scripts/run_phase68_multistrat_live.py` (upgraded), `src/rde/trading/carry_executor.py`, `docs/findings/phase77_multistrat_resheet.md` | Multi-strategy re-tearsheet with Phase 75+76 carry upgrades. **STRONG GO.** Carry Sharpe 19.73 (from 18.13). Combined 80/15/5 Sharpe **11.9359** (+0.89 vs Phase 71). MDD −0.21% → **−0.10%**. Every year positive, 2022 stress +8.00%. `make multistrat-backtest`. |
| 78 | `scripts/run_phase78_carry_stress.py`, `docs/findings/phase78_carry_stress.md` | Carry MDD robustness stress test. **ROBUST.** Near-zero MDD (−0.02%) is genuine, not path-specific. Half A Sharpe 22.60, Half B Sharpe 28.86 (both > 15.0 threshold). Permutation p=0.0000 (0/100 shuffles beat real Sharpe of 19.69 — real Sharpe +1.64 above shuffled mean). Momentum filter exits before worst period in 9 of 10 stress episodes; worst unavoidable first-period loss = −244% ann (May 2021 flash crash, 1 period). `make carry-phase78`. Updated `carry-live` defaults to include `--momentum-filter --spot-spread-weight`. Added `carry-live-full` target. |

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

**Phase 51b GO — spy_rank_bull deployed.** `lambda_by_state_rank=[0.02, 0.05, 0.10]` with `lambda_proxy_asset="SPY"`. Sharpe=0.8948 (+0.011 vs fixed_l05 baseline of 0.8838). Marginal pass — monitor over live paper trading period.

**Phase 52a GO — 5-asset universe (SPY/GLD/SHY/IEF/TLT) replaces 4-asset.** spy_rank_bull Sharpe=0.9726 (+0.0778 vs 4-asset baseline). MDD −15.6% (was −21.3%). Update live config to use the 5-asset universe. 6-asset (+ TIP) is marginal NO relative to 5-asset (lower Sharpe −0.006, though lower MDD −0.7pp).

**Phase 53 NO-GO — online-posterior KL monitor failed.** All thresholds [0.05–2.0] fire at 45.7 triggers/year regardless of KL threshold — `kl_min_bars_between_triggers=5` is the binding constraint. Forward filter on n=3 daily HMM oscillates between states every ~5 bars; KL is always large relative to any tested threshold. Confidence gate (`kl_min_dominant_confidence=0.50`) did not help. Root cause: online forward-filter posteriors are inherently noisy on daily data. No threshold produces ≤4 triggers/year. See `docs/findings/phase53_kl_monitor.md`.

**Completed research stack (all validated):**

- **Phase 57 + Phase 63 DONE — STRONG GO.** momentum_tilt_scale=0.03 achieves Sharpe 1.0008 (+0.028).
  Phase 63 CV: fold=3/5 PASS, break-even=∞ PASS, shuffle p=**0.0000** (strongest possible, 0/50 shuffles beat real).
  `momentum_tilt_scale=0.03` is the confirmed live baseline.

- **Phase 60 + Phase 62 DONE — STRONG GO.** carry_weighted Sharpe 18.129. Bull-only scale (+3.4% ann).

- **Phase 65 + Phase 67 DONE — STRONG GO.** LETF pair Sharpe 4.887, every year positive, live executor operational.

- **Phase 66 DONE.** Portfolio frontier: **80% carry + 15% LETF pair + 5% RTMV** (Sharpe 9.828, MDD −0.55%).

- **Phase 61 DONE.** Carry dominates combined Sharpe. Pairwise correlations all near-zero (~−0.01).

- **Phase 70 DONE — PARTIAL GO.** 3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY) Sharpe 5.47 > 5.0 threshold. Individual pairs all NO-GO (≤4.58). Combo is the clear winner: diversification reduces MDD from −2.04% to −1.74%.

- **Phase 71 DONE — STRONG GO.** Combined multi-strategy Sharpe 11.5636 (+0.515 vs Phase 68 baseline 11.0483). MDD −0.20%. Every year positive. `scripts/run_phase68_multistrat_live.py` now uses 3-pair LETF combo.

- **Phase 72 DONE — NO-GO (hypothesis confirmed).** SPY HMM regime filter on 3-pair LETF combo. Bear regime LETF ann return=61.02%, neutral=20.26%, bull=8.91% — LETF earns most from VOLATILITY, not direction. Bull-only 15× Sharpe=5.80 (vs flat 5.47). Does not clear 7.06 GO threshold. Always-in flat remains the optimal LETF configuration. `make letf-regime-filter`.

- **Phase 73 DONE — CONDITIONAL GO (carry-basket NO-GO).** SOL perpetual funding carry re-test with 2024-2026 data. SOL 2024+ mean carry 5.3% (improved), but SOL/ETH correlation jumped to 0.695 (was 0.258 in Phase 59) — diversification benefit eliminated. Best SOL basket Sharpe 17.601 vs Phase 62 live baseline 17.943 (delta -0.342). BTC+ETH basket remains optimal. `make carry-phase73`.

- **Phase 74 DONE — NO-GO.** Portfolio frontier re-optimisation with upgraded 3-pair LETF leg. Fine grid (63 combinations, carry=[50..90%], letf=[5..35%], rtmv=[2..15%]). Best: 75/20/5 Sharpe 11.5794 — improvement only +0.0158 vs +0.09 GO threshold. 80/15/5 remains optimal. Root cause: carry (Sharpe 8.96) dominates; LETF vol (5.20%/yr) dilutes Sharpe at higher weights despite carry↔LETF=−0.097 diversification benefit. `make portfolio-reopt`.

- **Phase 75 DONE — GO (lag-1 momentum filter).** Intraday UTC window filtering: **NULL RESULT** — skipping any fixed window (00:00/08:00/16:00) costs 4.1–4.3 Sharpe (33% fewer compounding periods). Window carry differences are tiny (12.7% vs 13.6% combined, only 0.9pp). **Lag-1 momentum filter: GO** — carry exhibits very strong autocorrelation (lag-1=0.793, lag-3=0.675). Skipping current period when previous combined rate ≤ 0 improves Sharpe 18.1294 → **18.5728** (+0.4434), Ann Return 14.13% → 14.19%, MDD −0.46% → **−0.22%**. Root cause: the filter removes ~16% of periods where carry turns negative — exactly the drawdown-generating periods — without sacrificing return. `make carry-phase75`. See `docs/findings/phase75_intraday_carry.md`.

- **Phase 76 DONE — STRONG GO (Part A deployed + Part B GO).** **Part A:** Lag-1 momentum filter deployed to `CarryStrategy(momentum_filter=True)` (`src/rde/trading/carry_executor.py`). `--momentum-filter` CLI flag added to `run_carry_live.py`. `make carry-live-filtered` target added for filtered live mode. Verified: filtered baseline Sharpe = **18.5728** (matches Phase 75 +0.4434 improvement, MDD halved to −0.22%). **Part B:** Instantaneous ETH-BTC carry spread signal. **STRONG GO.** `spot_spread_dynamic` (ETH weight = min(0.80, max(0.40, ETH_rate/(ETH_rate+BTC_rate)))) Sharpe **19.7283** (+1.16 vs momentum-filtered baseline, +1.60 vs unfiltered). MDD **−0.02%** (near-zero). `spot_spread_15x` (ETH=75% when ETH/BTC > 1.5) Sharpe 19.08 (+0.51). Root cause of improvement: ETH is above 1.5× BTC carry for 21.2% of periods — during those periods proportional overweighting of ETH captures a structural premium that rolling 90-period weights smooth away. `make carry-phase76`. See `docs/findings/phase76_carry_spread.md`.

- **Phase 77 DONE — STRONG GO.** `spot_spread_weight: bool = False` and `spot_spread_min/max` parameters added to `CarryStrategy`. `--spot-spread-weight` CLI flag added to `run_carry_live.py`. Carry leg Sharpe progression: 18.13 (Phase 60) → 18.57 (Phase 75 filter) → **19.73** (Phase 76 spread). `build_carry_daily()` in `run_phase68_multistrat_live.py` updated to use both improvements. **Combined multi-strategy (80/15/5) Sharpe: 11.9359** (up from Phase 71 baseline 11.05, +0.89). MDD improves −0.21% → **−0.10%**. Every year positive including 2022 (+8.00%). Cross-strategy correlations unchanged (all |ρ| < 0.08). GO threshold 11.05 cleared by +0.89. `make multistrat-backtest`. See `docs/findings/phase77_multistrat_resheet.md`.

- **Phase 78 DONE — ROBUST.** Carry MDD stress test confirms near-zero MDD (−0.02%) is genuine. Half A Sharpe 22.60 (2019-2023), Half B Sharpe 28.86 (2023-2026) — both well above 15.0 threshold. Permutation p=**0.0000** (0/100 shuffles beat real Sharpe 19.69; shuffled mean 18.05, margin +1.64). Momentum filter exits before the worst period in 9 of 10 stress episodes. Only exception: May 2021 flash crash (1-period single-bar event, −244% ann first-period loss, unavoidable by any lag-1 filter). Top 3 episodes protected: Sep 2022 (saved −648.68% ann), Mar 2020 COVID (saved −406.57% ann), Nov 2022 FTX (saved −285.90% ann). `carry-live` defaults updated to `--momentum-filter --spot-spread-weight`. `carry-live-full` target added. `make carry-phase78`. See `docs/findings/phase78_carry_stress.md`.

- **Phase 79 DONE — Deployment hardening.** All live-mode Makefile targets verified and updated. `multistrat-signal` updated to show Phase 75-78 carry filter status (momentum filter active, spot-spread weight ratio, combined rate). `.PHONY` list updated. Deployment checklist written at `docs/DEPLOYMENT_CHECKLIST.md` covering all three legs with phase history, start/restart procedure, health check commands, and backtest benchmarks. Test suite: 1577 passing, 0 failures. `make multistrat-signal` confirmed operational — shows per-leg signals + Phase 76 ETH/BTC instantaneous weight + Phase 75 momentum filter confirmation + Phase 78 robustness tag.

- **Phase 80 DONE — Sharpe Attribution Audit.** Analytical audit of every documented research decision's contribution to combined portfolio Sharpe (Phase 66→77, 9.828→11.936). Key findings: (1) Signal design is highest ROI per phase — Phases 75+76 (carry momentum filter + spot_spread_dynamic) added +1.604 carry Sharpe and +0.372 combined with zero new data sources and 2 lines of code each. (2) Universe expansion is second — Phase 71 (3-pair LETF) added +0.516 combined. (3) Parameter tuning has diminishing returns (~+0.017 RTMV Sharpe per phase across 4 phases). (4) Structural changes had negative ROI — Phase 54 (joint HMM, highest complexity) was conditional NO-GO, −0.021 vs simpler baseline. Full table in `docs/findings/phase80_sharpe_attribution.md`. Script: `scripts/run_phase80_sharpe_attribution.py`.

- **Phase 81 DONE — RTMV Tilt P&L Lag-1 Autocorrelation. NO-GO (NULL RESULT).** Tested whether RTMV's per-rebalance regime-tilt P&L has lag-1 autocorrelation exploitable by a momentum filter (analogous to Phase 75 carry momentum filter). Key findings: lag-1 ACF = **−0.1504** (exactly at the ±0.15 threshold — borderline null result). Carry comparison: carry lag-1 ACF = 0.793 vs RTMV tilt = −0.1504. Root cause of the gap: carry rates are structurally persistent (supply/demand driven, 8-hour horizon); RTMV tilt P&L reflects monthly HMM direction prediction with 95% overlapping training windows — each month's misprediction is nearly independent of the next. 59.1% of periods tilt was positive (mean = +0.021% per period), confirming RTMV adds persistent but small edge. Baseline Sharpe = 1.0010, Pure GMV = 0.9909 (+0.010 tilt premium). Variants tested: lag1_filter (λ→0.02 after loss), lag1_reduce (λ×0.5 after loss), lag1_zero (λ=0 after loss). All variants designed for momentum; none match the borderline mean-reversion direction. GO threshold: Sharpe ≥ 1.030. See `docs/findings/phase81_rtmv_momentum.md`; script: `scripts/run_phase81_rtmv_momentum.py`.

**Next direction — Phase 82: Dynamic LETF allocation (bear-regime overweight).**
Phase 81 NULL RESULT closes the RTMV signal-design path. The next highest available lever
from the Phase 80 attribution table is allocation-level dynamic weighting in the LETF leg.

Hypothesis: When SPY HMM rank = 0 (bear), increase the multi-strategy LETF weight from
15% → 25% (reduce carry from 80% → 70%). LETF earns +61% ann in bear regimes vs +9% in
bull (Phase 70/72 finding). Phase 72 tested within-LETF position scaling (NO-GO). Phase 82
tests CROSS-STRATEGY allocation shifting — fundamentally different mechanism.

Key evidence from Phase 68 2022 stress test: carry +2.1%, LETF +44.3%, RTMV −12.2% →
combined +6.8%. Dynamically overweighting LETF in that regime would add ~4.4% to the
combined return (+44.3% × 10% increment) at minimal vol cost (carry/LETF corr = −0.084).

- GO threshold: combined portfolio Sharpe ≥ 12.0 (Phase 77 baseline 11.936 + 0.064)
- Implementation: modify `scripts/run_phase68_multistrat_live.py` with `--letf-bear-weight-factor`
  (default 1.0 = no change). When SPY HMM rank = 0 at month-end, multiply LETF allocation
  by factor and renormalise carry + LETF + RTMV proportionally.

**Alpha hunt synthesis:** `docs/findings/alpha_hunt_synthesis.md` — complete ranking
of all confirmed and pending strategies with capital allocation percentages.

The Notion roadmap and tasks database are the source of truth for sequencing. Update Notion as work progresses.

---

## 10. Working with Claude Code

- **Read this file first.** It is the project's invariants.
- **When making architectural decisions, propose them in the chat, then add an entry to the Decision Log in Notion before implementing.** A code change that contradicts CLAUDE.md without an updated CLAUDE.md is a bug.
- **When proposing new features that go beyond v1.0**, flag them as post-v1.0 and add them to the Notion parking lot rather than implementing.
- **When uncertain, say so explicitly.** Do not guess at numerical thresholds, paper conventions, or library behavior. Verify or ask.
- **When fixing existing notebooks** (the project knowledge contains a working but limited reference notebook), treat them as informational, not prescriptive. The new system supersedes them.
