"""RTMV live rebalancer for the SPY/GLD/TLT/IEF portfolio (Phase 48).

Wraps :func:`rde.analysis.multi_asset_allocation.compute_rtmv_weights_now`
and :class:`MultiAssetPortfolio` into a daily-polling, monthly-rebalancing
deployment skeleton.

Usage modes
-----------
*Backtest* (default in CLI): feed pre-downloaded ``asset_returns`` and
``asset_features`` DataFrames bar-by-bar via :meth:`RTMVRebalancer.run_backtest`.

*Live* (polling): :meth:`RTMVRebalancer.run_forever` polls yfinance every
``config.poll_interval_s`` seconds, finds new bars, and calls
:meth:`RTMVRebalancer.step` for each.

Public API
----------
RTMVRebalancerConfig
RTMVRebalancerState
RTMVRebalancer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RTMVRebalancerConfig:
    """Configuration for :class:`RTMVRebalancer`.

    Attributes
    ----------
    assets : list[str]
        Ticker symbols in portfolio order, e.g. ``["SPY", "GLD", "TLT", "IEF"]``.
    lambda_tilt : float
        RTMV blending weight in [0, 1].  Phase 47 CONDITIONAL GO lambda.
    rebalance_bars : int
        Rebalance every N bars (21 ≈ one calendar month for daily data).
    lookback_bars : int
        Training window length for per-asset HMM fits.
    n_states : int
        HMM state count.  Phase 43c optimum for this universe.
    n_restarts : int
        HMM training restarts.
    initial_capital : float
    slippage_bps : float
        One-way execution slippage.
    drawdown_halt : float
        Halt trading if portfolio peak-to-trough drawdown exceeds this
        fraction.  Rebalances are skipped (hold current weights) until
        equity recovers above the previous peak.
    poll_interval_s : float
        Seconds between yfinance polls in live mode.
    output_dir : Path
        Directory for snapshots and fills parquet files.
    rebalance_kl_threshold : float
        Posterior KL-divergence threshold for early rebalancing (Phase 50b).
        At each bar, the rebalancer computes the mean across assets of
        ``KL(current_posterior || posterior_at_last_rebalance)``.  When that
        mean exceeds this threshold, an early rebalance is triggered (in
        addition to the calendar trigger ``rebalance_bars``).  Set to
        ``0.0`` (default) to disable — calendar-only behaviour.

        Note: enabling this requires fitting per-asset HMMs every bar to
        obtain current posteriors, which is significantly more expensive
        than calendar-only mode.
    """

    assets: list[str] = field(default_factory=lambda: ["SPY", "GLD", "TLT", "IEF"])
    lambda_tilt: float = 0.05
    rebalance_bars: int = 21
    lookback_bars: int = 504
    n_states: int = 3
    n_restarts: int = 3
    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    drawdown_halt: float = 0.25
    poll_interval_s: float = 3600.0
    output_dir: Path = Path("results/rtmv_live")
    adaptive_lambda: bool = False
    lambda_min: float = 0.02
    lambda_max: float = 0.15
    rebalance_kl_threshold: float = 0.0
    lambda_by_state_rank: list[float] = field(default_factory=list)
    lambda_proxy_asset: str | None = None
    kl_trigger_threshold: float = 0.0
    kl_min_bars_between_triggers: int = 5


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class RTMVRebalancerState:
    """Mutable runtime state exposed for inspection.

    Attributes
    ----------
    n_bars_processed : int
    n_rebalances : int
    last_date : pd.Timestamp or None
    equity : float
    current_weights : dict[str, float]
    peak_equity : float
        High-water mark for drawdown monitoring.
    is_halted : bool
        True when drawdown > ``config.drawdown_halt``.
    halt_reason : str
    snapshots : list[dict]
        Per-bar state records (written to parquet on save).
    posterior_at_last_rebalance : dict
        Maps asset name → ``np.ndarray`` of shape ``(n_states,)``: the
        last-bar HMM filtered posterior captured at the most recent
        rebalance.  Empty dict before the first rebalance.  Used by
        Phase 50b posterior-triggered early rebalancing.
    n_kl_rebalances : int
        Number of rebalances triggered by KL divergence (subset of
        ``n_rebalances``).  Useful for diagnosing trigger frequency.
    """

    n_bars_processed: int = 0
    n_rebalances: int = 0
    last_date: pd.Timestamp | None = None
    equity: float = 0.0
    current_weights: dict = field(default_factory=dict)
    peak_equity: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    snapshots: list[dict] = field(default_factory=list)
    posterior_at_last_rebalance: dict = field(default_factory=dict)
    n_kl_rebalances: int = 0
    last_lambda_effective: float = 0.05
    # Phase 53: reference-posterior KL monitor
    kl53_reference_posteriors: dict = field(default_factory=dict)
    kl53_online_decoders: dict = field(default_factory=dict)
    kl53_bars_since_trigger: int = 0
    n_kl53_triggers: int = 0


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------


class RTMVRebalancer:
    """Daily-polling, monthly-rebalancing RTMV portfolio manager.

    Parameters
    ----------
    config : RTMVRebalancerConfig
    ma_config : MultiAssetConfig, optional
        Passed directly to :func:`compute_rtmv_weights_now`.  Defaults to
        a ``MultiAssetConfig`` built from ``config`` fields.
    train_kwargs : dict, optional
        Forwarded to :func:`rde.models.hmm.train_hmm` at each rebalance.
    """

    def __init__(
        self,
        config: RTMVRebalancerConfig | None = None,
        ma_config=None,
        train_kwargs: dict | None = None,
        supabase_writer=None,
    ) -> None:
        from rde.analysis.multi_asset_allocation import MultiAssetConfig
        from rde.trading.multi_asset_portfolio import (
            MultiAssetPortfolio,
            MultiAssetPortfolioConfig,
        )

        self.config = config or RTMVRebalancerConfig()
        self._train_kwargs = train_kwargs or {}

        # Build MultiAssetConfig from rebalancer config.
        if ma_config is None:
            self.ma_config = MultiAssetConfig(
                ann_factor=252,
                rebalance_bars=self.config.rebalance_bars,
                lookback_bars=self.config.lookback_bars,
                n_states=self.config.n_states,
                n_restarts=self.config.n_restarts,
                transaction_cost=self.config.slippage_bps / 10_000,
            )
        else:
            self.ma_config = ma_config

        self.portfolio = MultiAssetPortfolio(
            MultiAssetPortfolioConfig(
                initial_capital=self.config.initial_capital,
                slippage_bps=self.config.slippage_bps,
            ),
            self.config.assets,
        )

        self.state = RTMVRebalancerState(
            equity=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
        )

        # Internal rolling buffers: index = date, columns = assets/features.
        self._returns_buf: pd.DataFrame = pd.DataFrame(
            columns=self.config.assets
        )
        # Per-asset feature buffers.
        self._features_buf: dict[str, pd.DataFrame] = {
            a: pd.DataFrame() for a in self.config.assets
        }
        self._bars_since_rebalance: int = 0
        self._supabase = supabase_writer  # SupabaseWriter | None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_buffers(
        self,
        date: pd.Timestamp,
        returns_row: dict[str, float],
        features_rows: dict[str, dict[str, float]],
    ) -> None:
        """Append one bar to the internal rolling buffers."""
        ret_df = pd.DataFrame([returns_row], index=[date])
        self._returns_buf = pd.concat([self._returns_buf, ret_df])

        for asset in self.config.assets:
            if asset in features_rows and features_rows[asset]:
                feat_df = pd.DataFrame([features_rows[asset]], index=[date])
                self._features_buf[asset] = pd.concat(
                    [self._features_buf[asset], feat_df]
                )

        # Trim to lookback_bars + a small buffer to avoid memory growth.
        trim = self.config.lookback_bars + 50
        if len(self._returns_buf) > trim:
            self._returns_buf = self._returns_buf.iloc[-trim:]
        for asset in self.config.assets:
            if len(self._features_buf[asset]) > trim:
                self._features_buf[asset] = self._features_buf[asset].iloc[-trim:]

    def _current_drawdown(self) -> float:
        """Current peak-to-trough drawdown as a positive fraction."""
        eq = self.state.equity
        peak = self.state.peak_equity
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - eq) / peak)

    def _check_drawdown_halt(self) -> bool:
        """Return True if trading should proceed (not halted)."""
        dd = self._current_drawdown()
        if dd > self.config.drawdown_halt:
            if not self.state.is_halted:
                self.state.is_halted = True
                self.state.halt_reason = (
                    f"Drawdown {dd:.1%} exceeds limit {self.config.drawdown_halt:.1%}"
                )
                logger.warning("RTMV HALT: %s", self.state.halt_reason)
            return False

        # Recover if we were halted and equity is back above peak.
        if self.state.is_halted and self.state.equity >= self.state.peak_equity:
            self.state.is_halted = False
            self.state.halt_reason = ""
            logger.info("RTMV: drawdown halt cleared — equity recovered to peak.")

        return not self.state.is_halted

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        date: pd.Timestamp,
        prices: dict[str, float],
        returns_row: dict[str, float],
        features_rows: dict[str, dict[str, float]],
    ) -> float:
        """Process one bar.

        Updates internal buffers, optionally triggers a rebalance, marks
        the portfolio to market, and records a snapshot.

        Parameters
        ----------
        date : pd.Timestamp
        prices : dict[str, float]
            Current close prices per asset.
        returns_row : dict[str, float]
            Log returns for this bar per asset.
        features_rows : dict[str, dict[str, float]]
            Feature values per asset for this bar.
            Keys are asset names; values are dicts of feature→value.

        Returns
        -------
        float
            Current portfolio equity.
        """
        from rde.analysis.multi_asset_allocation import compute_rtmv_weights_now

        self._update_buffers(date, returns_row, features_rows)
        self._bars_since_rebalance += 1
        self.state.n_bars_processed += 1
        self.state.last_date = date

        # Mark to market first.
        eq = self.portfolio.equity(prices)
        self.state.equity = eq
        if eq > self.state.peak_equity:
            self.state.peak_equity = eq

        # Decide on rebalance.
        enough_history = len(self._returns_buf) >= self.ma_config.lookback_bars
        calendar_due = self._bars_since_rebalance >= self.config.rebalance_bars
        trading_ok = self._check_drawdown_halt()

        target_weights: dict[str, float] | None = None
        triggered_by_kl = False
        mean_kl: float | None = None
        triggered_by_kl53 = False
        mean_kl53: float | None = None
        self.state.kl53_bars_since_trigger += 1

        # Phase 53: cheap online-posterior KL check (no refit between rebalances).
        # Requires kl53 reference decoders to exist (set at last calendar rebalance).
        kl53_check_active = (
            self.config.kl_trigger_threshold > 0.0
            and self.state.kl53_online_decoders
            and enough_history
            and trading_ok
            and not calendar_due
            and self.state.kl53_bars_since_trigger >= self.config.kl_min_bars_between_triggers
        )
        if kl53_check_active:
            online_posts: dict[str, np.ndarray] = {}
            for asset in self.config.assets:
                dec = self.state.kl53_online_decoders.get(asset)
                if dec is None:
                    continue
                feat_buf = self._features_buf[asset]
                if feat_buf.empty or asset not in features_rows:
                    continue
                cols = list(feat_buf.columns)
                x = np.array([features_rows[asset].get(c, 0.0) for c in cols], dtype=float)
                online_posts[asset] = dec.step(x)

            if online_posts and self.state.kl53_reference_posteriors:
                mean_kl53 = self._mean_kl_53(online_posts)
                dominant_changed = any(
                    int(np.argmax(online_posts[a]))
                    != int(np.argmax(self.state.kl53_reference_posteriors[a]))
                    for a in online_posts
                    if a in self.state.kl53_reference_posteriors
                )
                if mean_kl53 > self.config.kl_trigger_threshold and dominant_changed:
                    ret_window = self._returns_buf.tail(self.ma_config.lookback_bars)
                    feat_window = {
                        a: self._features_buf[a].tail(self.ma_config.lookback_bars)
                        for a in self.config.assets
                    }
                    result = compute_rtmv_weights_now(
                        ret_window,
                        feat_window,
                        config=self.ma_config,
                        lambda_tilt=self.config.lambda_tilt,
                        train_kwargs=self._train_kwargs,
                        return_posteriors=False,
                        return_models=False,
                        lambda_by_state_rank=self.config.lambda_by_state_rank or None,
                        lambda_proxy_asset=self.config.lambda_proxy_asset,
                    )
                    target_weights = result
                    triggered_by_kl53 = True
                    self.state.n_kl53_triggers += 1
                    self.state.kl53_bars_since_trigger = 0

        # Phase 50b KL-triggered early rebalance (legacy, disabled by default).
        kl_check_active = (
            self.config.rebalance_kl_threshold > 0.0
            and len(self.state.posterior_at_last_rebalance) > 0
            and enough_history
            and trading_ok
            and not calendar_due
            and target_weights is None
        )
        if kl_check_active:
            ret_window = self._returns_buf.tail(self.ma_config.lookback_bars)
            feat_window = {
                a: self._features_buf[a].tail(self.ma_config.lookback_bars)
                for a in self.config.assets
            }
            candidate_weights, current_posteriors = compute_rtmv_weights_now(
                ret_window,
                feat_window,
                config=self.ma_config,
                lambda_tilt=self.config.lambda_tilt,
                train_kwargs=self._train_kwargs,
                adaptive_lambda=self.config.adaptive_lambda,
                lambda_min=self.config.lambda_min,
                lambda_max=self.config.lambda_max,
                return_posteriors=True,
                lambda_by_state_rank=self.config.lambda_by_state_rank or None,
                lambda_proxy_asset=self.config.lambda_proxy_asset,
            )
            mean_kl = self._mean_posterior_kl(current_posteriors)
            if mean_kl > self.config.rebalance_kl_threshold:
                target_weights = candidate_weights
                self.state.posterior_at_last_rebalance = current_posteriors
                triggered_by_kl = True

        if target_weights is None and enough_history and calendar_due and trading_ok:
            ret_window = self._returns_buf.tail(self.ma_config.lookback_bars)
            feat_window = {
                a: self._features_buf[a].tail(self.ma_config.lookback_bars)
                for a in self.config.assets
            }
            need_posteriors = self.config.rebalance_kl_threshold > 0.0
            need_models = self.config.kl_trigger_threshold > 0.0
            result = compute_rtmv_weights_now(
                ret_window,
                feat_window,
                config=self.ma_config,
                lambda_tilt=self.config.lambda_tilt,
                train_kwargs=self._train_kwargs,
                adaptive_lambda=self.config.adaptive_lambda,
                lambda_min=self.config.lambda_min,
                lambda_max=self.config.lambda_max,
                return_posteriors=need_posteriors or need_models,
                return_models=need_models,
                lambda_by_state_rank=self.config.lambda_by_state_rank or None,
                lambda_proxy_asset=self.config.lambda_proxy_asset,
            )
            if need_models:
                target_weights, current_posteriors, fitted_models = result
                self.state.posterior_at_last_rebalance = current_posteriors
                # Phase 53: initialise online decoders from new fitted models.
                from rde.inference.online import OnlineDecoder
                new_decoders: dict = {}
                new_ref_posts: dict = {}
                for asset, fitted in fitted_models.items():
                    feat_arr = feat_window[asset].values.astype(float)
                    dec = OnlineDecoder(fitted)
                    dec.batch_filter(feat_arr)  # warm up to last bar of lookback
                    new_decoders[asset] = dec
                    new_ref_posts[asset] = dec.filtered_posterior.copy()
                self.state.kl53_online_decoders = new_decoders
                self.state.kl53_reference_posteriors = new_ref_posts
                self.state.kl53_bars_since_trigger = 0
            elif need_posteriors:
                target_weights, current_posteriors = result
                self.state.posterior_at_last_rebalance = current_posteriors
            else:
                target_weights = result

        if target_weights is not None:
            new_fills = self.portfolio.set_target_weights(target_weights, prices, date)
            self._bars_since_rebalance = 0
            self.state.n_rebalances += 1
            if triggered_by_kl:
                self.state.n_kl_rebalances += 1
            self.state.current_weights = dict(target_weights)
            self.state.last_lambda_effective = self.config.lambda_tilt

            if self._supabase is not None and new_fills:
                fills_rows = [
                    {
                        "fill_at": f.timestamp.isoformat() if hasattr(f.timestamp, "isoformat") else str(f.timestamp),
                        "asset": f.asset,
                        "side": f.side,
                        "quantity": round(float(f.quantity), 6),
                        "price": round(float(f.price), 6),
                        "slippage": round(float(f.slippage), 6),
                        "notional": round(float(f.notional), 4),
                    }
                    for f in new_fills
                ]
                self._supabase.write_fills(fills_rows)

            eq = self.portfolio.equity(prices)
            self.state.equity = eq
            if eq > self.state.peak_equity:
                self.state.peak_equity = eq

            trigger_reason = "KL53" if triggered_by_kl53 else ("KL" if triggered_by_kl else "calendar")
            logger.info(
                "%s  REBALANCE #%d (%s)  weights=%s  equity=%.2f",
                date.date() if hasattr(date, "date") else date,
                self.state.n_rebalances,
                trigger_reason,
                {k: f"{v:.3f}" for k, v in target_weights.items()},
                eq,
            )

        snap = self.portfolio.snapshot(date, prices, target_weights) | {
            "n_rebalances": self.state.n_rebalances,
            "drawdown": self._current_drawdown(),
            "is_halted": self.state.is_halted,
            "rebalance": target_weights is not None,
            "kl_triggered": triggered_by_kl,
            "mean_kl": mean_kl if mean_kl is not None else float("nan"),
            "kl53_triggered": triggered_by_kl53,
            "mean_kl53": mean_kl53 if mean_kl53 is not None else float("nan"),
            "n_kl53_triggers": self.state.n_kl53_triggers,
        }
        self.state.snapshots.append(snap)

        # Optional Supabase write (no-op if writer is None or disabled).
        if self._supabase is not None:
            bar_date = date.date() if hasattr(date, "date") else date
            weight_cols = {k[len("weight_"):]: v for k, v in snap.items() if k.startswith("weight_")}
            target_cols = {k[len("target_"):]: v for k, v in snap.items() if k.startswith("target_")}
            self._supabase.write_snapshot(
                bar_date=bar_date,
                equity=float(snap.get("equity", 0.0)),
                cash=float(snap.get("cash", 0.0)),
                drawdown=float(snap.get("drawdown", 0.0)),
                peak_equity=float(self.state.peak_equity),
                is_halted=bool(self.state.is_halted),
                rebalanced=target_weights is not None,
                n_rebalances=int(self.state.n_rebalances),
                weights=weight_cols,
                target_weights=target_cols or None,
            )

        return self.state.equity

    # ------------------------------------------------------------------
    # KL helpers
    # ------------------------------------------------------------------

    def _mean_kl_53(self, online_posts: dict[str, np.ndarray]) -> float:
        """Mean across assets of KL(online_posterior || kl53_reference_posterior).

        Uses the same model for both sides — so KL reflects regime drift only,
        not model drift (the Phase 50b failure mode).
        """
        ref = self.state.kl53_reference_posteriors
        eps = 1e-15
        kls: list[float] = []
        for asset, p in online_posts.items():
            q = ref.get(asset)
            if q is None or p.shape != q.shape:
                continue
            p_safe = np.asarray(p, dtype=float) + eps
            q_safe = np.asarray(q, dtype=float) + eps
            p_safe = p_safe / p_safe.sum()
            q_safe = q_safe / q_safe.sum()
            kls.append(max(float(np.sum(p_safe * np.log(p_safe / q_safe))), 0.0))
        return float(np.mean(kls)) if kls else 0.0

    def _mean_posterior_kl(self, current: dict[str, np.ndarray]) -> float:
        """Mean across assets of KL(current || posterior_at_last_rebalance).

        Uses ``sum(p * log((p + eps) / (q + eps)))`` with eps=1e-15 for
        numerical safety.  Assets missing from either side are skipped; if
        no overlap exists, returns 0.0.
        """
        stored = self.state.posterior_at_last_rebalance
        eps = 1e-15
        kls: list[float] = []
        for asset, p in current.items():
            q = stored.get(asset)
            if q is None or p.shape != q.shape:
                continue
            p_safe = np.asarray(p, dtype=float) + eps
            q_safe = np.asarray(q, dtype=float) + eps
            p_safe = p_safe / p_safe.sum()
            q_safe = q_safe / q_safe.sum()
            kl = float(np.sum(p_safe * np.log(p_safe / q_safe)))
            kls.append(max(kl, 0.0))
        if not kls:
            return 0.0
        return float(np.mean(kls))

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        asset_returns: pd.DataFrame,
        asset_features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Backtest by processing all bars in chronological order.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            Shape (T, N).  Log returns aligned to a shared DatetimeIndex.
        asset_features : dict[str, pd.DataFrame]
            Per-asset feature DataFrames aligned to the same index.

        Returns
        -------
        pd.DataFrame
            Snapshot DataFrame indexed by date with equity, weights,
            rebalance flags, and drawdown.
        """
        assets = self.config.assets
        # Build price series from cumulative returns (for mark-to-market).
        # Prices are rebased to 100.0 at start so position quantities are
        # denominated consistently across assets.
        prices_df = (1.0 + asset_returns).cumprod() * 100.0

        # Seed the portfolio with equal-weight allocation on bar 0.
        # This matches the Phase 45 research convention where weight_arr
        # starts as 1/N for every bar, ensuring returns are earned from
        # the first bar rather than sitting in cash during warm-up.
        first_date = asset_returns.index[0]
        first_prices = {a: float(prices_df.iloc[0][a]) for a in assets if a in prices_df.columns}
        equal_weights = {a: 1.0 / len(assets) for a in assets}
        self.portfolio.set_target_weights(equal_weights, first_prices, first_date)
        self.state.current_weights = equal_weights

        for date, ret_row in asset_returns.iterrows():
            prices = {a: float(prices_df.loc[date, a]) for a in assets if a in prices_df.columns}
            returns_row = {a: float(ret_row[a]) for a in assets if a in ret_row.index}
            features_rows = {}
            for a in assets:
                feat_df = asset_features.get(a)
                if feat_df is not None and date in feat_df.index:
                    features_rows[a] = {c: float(feat_df.loc[date, c]) for c in feat_df.columns}

            self.step(date, prices, returns_row, features_rows)

        snaps = self.snapshots_df()
        if self._supabase is not None and not snaps.empty:
            eq = snaps["equity"].dropna()
            ret = eq.pct_change().dropna()
            ann = 252
            mu = float(ret.mean()) * ann
            sig = float(ret.std()) * (ann ** 0.5) + 1e-15
            sharpe = mu / sig
            peak = eq.cummax()
            mdd = float(((eq - peak) / (peak + 1e-15)).min())
            self._supabase.finalize_run(sharpe, mdd, float(eq.iloc[-1]))
        return snaps

    # ------------------------------------------------------------------
    # Live polling mode
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Poll yfinance for new daily bars and process them.

        Runs until interrupted by Ctrl-C.  Saves state every rebalance.
        """
        import time

        import yfinance as yf
        from rde.features.pipeline import FeaturePipeline
        from rde.features.returns import LogReturns, SmoothedReturns
        from rde.features.volatility import RollingVolatility

        pipeline = FeaturePipeline([
            LogReturns(),
            RollingVolatility(window=20),
            SmoothedReturns(window=5),
        ])

        last_date: pd.Timestamp | None = None
        logger.info(
            "RTMV live loop started — assets=%s  λ=%.2f  poll=%.0fs",
            self.config.assets,
            self.config.lambda_tilt,
            self.config.poll_interval_s,
        )

        while True:
            try:
                raw = yf.download(
                    self.config.assets,
                    period="max",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    logger.warning("yfinance returned empty data — retrying.")
                    time.sleep(self.config.poll_interval_s)
                    continue

                if isinstance(raw.columns, pd.MultiIndex):
                    close_df = raw["Close"].copy()
                else:
                    close_df = raw[["Close"]].copy()

                # Build per-asset features.
                feat_dfs: dict[str, pd.DataFrame] = {}
                for asset in self.config.assets:
                    if asset not in close_df.columns:
                        continue
                    asset_raw = close_df[[asset]].rename(columns={asset: "Close"})
                    asset_raw["Open"] = asset_raw["Close"]
                    feat_df = pipeline.transform(asset_raw).dropna()
                    feat_dfs[asset] = feat_df

                # Align to common dates.
                common_dates = close_df.dropna().index
                for a in feat_dfs:
                    common_dates = common_dates.intersection(feat_dfs[a].index)

                new_dates = (
                    common_dates[common_dates > last_date]
                    if last_date is not None
                    else common_dates[-(self.config.lookback_bars + self.config.rebalance_bars + 5):]
                )

                for date in new_dates:
                    prices = {a: float(close_df.loc[date, a]) for a in self.config.assets if a in close_df.columns}
                    returns_row = {}
                    features_rows = {}
                    for a in self.config.assets:
                        if a in feat_dfs and date in feat_dfs[a].index:
                            row = feat_dfs[a].loc[date]
                            returns_row[a] = float(row.get("log_return", 0.0))
                            features_rows[a] = {c: float(row[c]) for c in feat_dfs[a].columns}
                    self.step(date, prices, returns_row, features_rows)
                    last_date = date

                logger.info(
                    "%s — equity=%.2f  rebalances=%d  halted=%s",
                    pd.Timestamp.now(tz="UTC").date(),
                    self.state.equity,
                    self.state.n_rebalances,
                    self.state.is_halted,
                )

            except KeyboardInterrupt:
                logger.info("Live loop stopped by user.")
                self.save_state()
                break
            except Exception as exc:
                logger.error("Live loop error: %s — retrying in %.0fs", exc, self.config.poll_interval_s)

            time.sleep(self.config.poll_interval_s)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def snapshots_df(self) -> pd.DataFrame:
        """Return the accumulated snapshots as a DataFrame indexed by date."""
        if not self.state.snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.state.snapshots)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df

    def save_state(self) -> None:
        """Write snapshots and fills to parquet in ``config.output_dir``."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        snaps = self.snapshots_df()
        if not snaps.empty:
            snaps.to_parquet(self.config.output_dir / "snapshots.parquet")
        self.portfolio.save_fills(self.config.output_dir / "fills.parquet")
        logger.info(
            "State saved: %d snapshots, %d fills → %s",
            len(snaps),
            len(self.portfolio.fills),
            self.config.output_dir,
        )
