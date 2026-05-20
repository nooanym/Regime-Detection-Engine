"""Crypto funding carry paper-trading executor (Phase 54).

Delta-neutral strategy: long spot + short perpetual futures on Binance.
Collects the 8-hourly funding payment when longs pay shorts (positive rate).

This module provides paper-trading simulation only — no real orders are placed.
Pattern mirrors MockExchange / MultiAssetPortfolio established in Phases 34/48.

Public API
----------
CarryPosition
CarryPortfolio
CarryStrategy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Binance USDM futures funding rate endpoint (public, no auth)
_FAPI_BASE = "https://fapi.binance.com/fapi/v1"
_PERIODS_PER_YEAR = 3 * 365  # 3 funding payments per day × 365


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CarryPosition:
    """A single delta-neutral carry position.

    Attributes
    ----------
    symbol : str
        Binance futures symbol, e.g. ``"ETHUSDT"``.
    spot_qty : float
        Units held in spot (long leg).
    perp_qty : float
        Units shorted in perpetual (short leg, same magnitude as spot_qty).
    entry_price : float
        Spot price at position open.
    entry_time : pd.Timestamp
        UTC timestamp of position entry.
    total_funding_collected : float
        Cumulative funding payments received (in rate units, not USD).
    """

    symbol: str
    spot_qty: float
    perp_qty: float
    entry_price: float
    entry_time: pd.Timestamp
    total_funding_collected: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class CarryPortfolio:
    """In-memory paper portfolio for delta-neutral funding carry positions.

    Tracks open positions and accumulates funding.  No exchange calls are made
    here — the caller supplies prices and funding rates at each step.

    Parameters
    ----------
    initial_capital : float
        Notional USD capital available (for tracking purposes only;
        position sizing is driven by qty passed to :meth:`enter`).
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self._initial_capital = initial_capital
        self._positions: dict[str, CarryPosition] = {}
        self._realized_pnl: float = 0.0
        self._funding_log: list[dict] = []

    @property
    def positions(self) -> dict[str, CarryPosition]:
        """Current open positions (shallow copy)."""
        return dict(self._positions)

    def enter(self, symbol: str, qty: float, spot_price: float) -> None:
        """Open a delta-neutral position for *symbol*.

        Long *qty* units of spot and short *qty* units of perp simultaneously.
        If a position already exists it is replaced (re-entry after exit).

        Parameters
        ----------
        symbol : str
        qty : float
            Number of units (always positive).
        spot_price : float
            Current spot price at entry.
        """
        if qty <= 0.0:
            raise ValueError(f"qty must be positive, got {qty}")
        if spot_price <= 0.0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        self._positions[symbol] = CarryPosition(
            symbol=symbol,
            spot_qty=qty,
            perp_qty=qty,
            entry_price=spot_price,
            entry_time=pd.Timestamp.now(tz="UTC"),
        )
        logger.info("ENTER carry %s qty=%.4f @ %.2f", symbol, qty, spot_price)

    def exit(self, symbol: str, spot_price: float) -> float:
        """Close the position for *symbol* and return realized P&L.

        P&L is the sum of:
        - Spot leg: (exit_price - entry_price) * qty  (long)
        - Perp leg: (entry_price - exit_price) * qty  (short, cancels spot)
        - Funding: total_funding_collected * entry_price * qty (approx USD value)

        The delta legs cancel so net P&L ≈ funding collected in USD terms.

        Parameters
        ----------
        symbol : str
        spot_price : float
            Current spot price at exit.

        Returns
        -------
        float
            Realized P&L in USD terms.

        Raises
        ------
        KeyError
            If no position is open for *symbol*.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")
        pos = self._positions.pop(symbol)
        spot_pnl = (spot_price - pos.entry_price) * pos.spot_qty
        perp_pnl = (pos.entry_price - spot_price) * pos.perp_qty
        funding_usd = pos.total_funding_collected * pos.entry_price * pos.spot_qty
        realized = spot_pnl + perp_pnl + funding_usd
        self._realized_pnl += realized
        logger.info(
            "EXIT carry %s qty=%.4f @ %.2f  pnl=%.2f USD (funding=%.2f)",
            symbol, pos.spot_qty, spot_price, realized, funding_usd,
        )
        return realized

    def update_funding(self, symbol: str, rate: float, timestamp: pd.Timestamp) -> None:
        """Apply a funding payment to an open position.

        Parameters
        ----------
        symbol : str
        rate : float
            Per-period funding rate (e.g. 0.0001 = 0.01%).  Positive means
            longs pay shorts — we collect as the short leg.
        timestamp : pd.Timestamp
            Time of the funding payment.
        """
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.total_funding_collected += rate
        ann = rate * _PERIODS_PER_YEAR
        self._funding_log.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "rate": rate,
            "ann_rate": ann,
            "cumulative": pos.total_funding_collected,
        })
        logger.debug("FUNDING %s rate=%.6f (%.2f%% ann) cum=%.4f", symbol, rate, ann * 100, pos.total_funding_collected)

    def snapshot(self, prices: dict[str, float] | None = None) -> dict:
        """Return a flat dict of portfolio state for logging.

        Parameters
        ----------
        prices : dict[str, float] or None
            Current spot prices per symbol.  Used for unrealized delta
            computation (should be ~0 for a delta-neutral book).

        Returns
        -------
        dict
        """
        prices = prices or {}
        total_funding = sum(p.total_funding_collected for p in self._positions.values())
        unrealized_delta = sum(
            (prices.get(sym, p.entry_price) - p.entry_price) * p.spot_qty
            for sym, p in self._positions.items()
        )
        return {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "open_positions": list(self._positions.keys()),
            "n_positions": len(self._positions),
            "total_funding_collected": total_funding,
            "unrealized_delta_usd": unrealized_delta,
            "realized_pnl_usd": self._realized_pnl,
        }

    def funding_log_df(self) -> pd.DataFrame:
        """Return the full funding log as a DataFrame."""
        if not self._funding_log:
            return pd.DataFrame(columns=["timestamp", "symbol", "rate", "ann_rate", "cumulative"])
        return pd.DataFrame(self._funding_log)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


def _fetch_latest_funding_rate(symbol: str, timeout: int = 10) -> float:
    """Fetch the most recent funding rate for *symbol* from Binance FAPI.

    Parameters
    ----------
    symbol : str
        Binance futures symbol, e.g. ``"ETHUSDT"``.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    float
        Latest funding rate (per 8-hour period).

    Raises
    ------
    RuntimeError
        If the fetch fails after SSL fallback.
    """
    import requests  # noqa: PLC0415

    url = f"{_FAPI_BASE}/fundingRate?symbol={symbol}&limit=1"
    try:
        r = requests.get(url, timeout=timeout, verify=True)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.SSLError:
        r = requests.get(url, timeout=timeout, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch funding rate for {symbol}: {exc}") from exc

    if not data:
        raise RuntimeError(f"Empty funding rate response for {symbol}")
    return float(data[-1]["fundingRate"])


class CarryStrategy:
    """Paper-trading carry strategy wrapper.

    Wraps :class:`CarryPortfolio` with entry/exit signal logic and
    persistence to parquet.  Supports both live polling and backtest replay.

    Parameters
    ----------
    symbols : list[str]
        Binance futures symbols to trade, e.g. ``["BTCUSDT", "ETHUSDT"]``.
    entry_threshold : float
        Minimum annualised carry to enter.  Default 5%.
    exit_threshold : float
        Annualised carry below which to exit.  Default -2%.
    qty_per_symbol : float
        Units to trade per symbol (paper only; no real sizing here).
    output_dir : Path or None
        Where to write parquet outputs.  Defaults to
        ``results/carry_live/``.
    initial_capital : float
        Passed to :class:`CarryPortfolio`.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        entry_threshold: float = 0.05,
        exit_threshold: float = -0.02,
        qty_per_symbol: float = 1.0,
        output_dir: Path | None = None,
        initial_capital: float = 100_000.0,
        regime_scale: dict[int, float] | None = None,
        carry_weighted: bool = True,
        carry_weight_window: int = 90,
        momentum_filter: bool = False,
        spot_spread_weight: bool = False,
        spot_spread_min: float = 0.40,
        spot_spread_max: float = 0.80,
    ) -> None:
        self._symbols: list[str] = symbols or ["BTCUSDT", "ETHUSDT"]
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._qty_per_symbol = qty_per_symbol
        self._output_dir = Path(output_dir or "results/carry_live")
        self._portfolio = CarryPortfolio(initial_capital=initial_capital)
        self._position_log: list[dict] = []
        # Regime scaling: maps SPY HMM dominant-state rank → position size multiplier.
        # bull-only (Phase 55 GO): {0: 1.0, 1: 1.0, 2: 1.5}
        self._regime_scale: dict[int, float] = regime_scale or {0: 1.0, 1: 1.0, 2: 1.5}
        # Phase 60 GO: weight symbols proportionally to trailing carry.
        # Requires ≥ carry_weight_window periods before carry weights are active.
        self._carry_weighted = carry_weighted
        self._carry_weight_window = carry_weight_window
        self._rate_history: dict[str, list[float]] = {sym: [] for sym in self._symbols}
        # Phase 75 GO: lag-1 momentum filter.
        # Skip periods when previous combined carry rate was ≤ 0.
        self._momentum_filter = momentum_filter
        self._last_combined_rate: float | None = None
        # Phase 76 GO: instantaneous spot-spread dynamic weighting (2-symbol case).
        # ETH weight = clip(eth_rate / (eth_rate + btc_rate), spot_spread_min, spot_spread_max).
        # Overrides carry_weighted rolling weights when True.
        self._spot_spread_weight = spot_spread_weight
        self._spot_spread_min = spot_spread_min
        self._spot_spread_max = spot_spread_max

    # ── Core step ────────────────────────────────────────────────────────────

    def step(
        self,
        rates: dict[str, float],
        prices: dict[str, float],
        timestamp: pd.Timestamp | None = None,
        hmm_rank: int | None = None,
    ) -> None:
        """Process one funding period.

        Parameters
        ----------
        rates : dict[str, float]
            Latest per-period funding rate per symbol.
        hmm_rank : int or None
            Dominant-state rank from the SPY HMM (0=bear, 1=neutral, 2=bull).
            When supplied, position size is scaled by ``regime_scale[hmm_rank]``.
        prices : dict[str, float]
            Current spot price per symbol (used for entry/exit and snapshot).
        timestamp : pd.Timestamp or None
            Timestamp for this step.  Defaults to UTC now.
        """
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        scale = self._regime_scale.get(hmm_rank, 1.0) if hmm_rank is not None else 1.0

        # Update rolling carry history for all symbols
        for sym in self._symbols:
            self._rate_history[sym].append(rates.get(sym, 0.0))

        # Compute combined portfolio-level carry rate (simple average across symbols).
        # Used by the Phase 75 lag-1 momentum filter.
        n_syms = len(self._symbols)
        combined_rate = sum(rates.get(sym, 0.0) for sym in self._symbols) / n_syms if n_syms > 0 else 0.0

        # Phase 75 momentum filter: skip period if previous combined rate was ≤ 0.
        # Hold cash — no positions entered/held, no funding collected this period.
        skip_period = (
            self._momentum_filter
            and self._last_combined_rate is not None
            and self._last_combined_rate <= 0
        )

        # Update lag state for next period *before* any early-return
        self._last_combined_rate = combined_rate

        # Compute per-symbol allocation weights.
        # Phase 76 spot_spread_weight (GO): instantaneous proportional weights for 2-symbol case.
        # Phase 60 carry_weighted (GO): rolling 90-period trailing carry proportional weights.
        # Default: equal weight.
        carry_weights: dict[str, float] = {}
        if self._spot_spread_weight and len(self._symbols) == 2:
            # Instantaneous proportional weight: w_sym1 = clip(rate1 / (rate1+ + rate2+), min, max)
            # Clip negative rates to 0 before proportional allocation.
            sym0, sym1 = self._symbols[0], self._symbols[1]
            r0 = max(0.0, rates.get(sym0, 0.0))
            r1 = max(0.0, rates.get(sym1, 0.0))
            denom = r0 + r1
            if denom > 0:
                raw_w1 = r1 / denom
                w1 = min(self._spot_spread_max, max(self._spot_spread_min, raw_w1))
                w0 = 1.0 - w1
            else:
                # Both rates ≤ 0: fall back to equal weight
                w0 = w1 = 0.5
            carry_weights = {sym0: w0, sym1: w1}
        elif self._carry_weighted and all(
            len(h) >= self._carry_weight_window for h in self._rate_history.values()
        ):
            trailing: dict[str, float] = {}
            for sym in self._symbols:
                hist = self._rate_history[sym][-self._carry_weight_window:]
                trailing[sym] = max(0.0, float(sum(hist) / len(hist)) * _PERIODS_PER_YEAR)
            total_carry = sum(trailing.values())
            if total_carry > 0:
                carry_weights = {sym: trailing[sym] / total_carry for sym in self._symbols}
            else:
                carry_weights = {sym: 1.0 / len(self._symbols) for sym in self._symbols}
        else:
            carry_weights = {sym: 1.0 / len(self._symbols) for sym in self._symbols}

        if skip_period:
            # Exit any open positions — hold cash for this period
            for sym in list(self._portfolio.positions.keys()):
                price = prices.get(sym, 1.0)
                self._portfolio.exit(sym, price)
        else:
            for sym in self._symbols:
                rate = rates.get(sym, 0.0)
                price = prices.get(sym, 0.0)
                ann_carry = rate * _PERIODS_PER_YEAR
                sym_qty = self._qty_per_symbol * scale * carry_weights[sym] * len(self._symbols)

                is_open = sym in self._portfolio.positions

                if not is_open and ann_carry > self._entry_threshold and sym_qty > 0:
                    self._portfolio.enter(sym, sym_qty, price)
                elif is_open and ann_carry < self._exit_threshold:
                    self._portfolio.exit(sym, price)
                elif is_open:
                    self._portfolio.update_funding(sym, rate, ts)

        snap = self._portfolio.snapshot(prices)
        snap["timestamp"] = ts
        snap["hmm_rank"] = hmm_rank
        snap["regime_scale"] = scale
        snap["carry_weights"] = str(carry_weights)
        snap["momentum_filter_skipped"] = skip_period
        self._position_log.append(snap)

    # ── Backtest mode ────────────────────────────────────────────────────────

    def run_backtest(
        self,
        funding_series: dict[str, pd.Series],
        prices: dict[str, pd.Series] | None = None,
        hmm_ranks: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Replay historical funding rates bar-by-bar.

        Parameters
        ----------
        funding_series : dict[str, pd.Series]
            Funding rate series per symbol, indexed by UTC timestamps.
        prices : dict[str, pd.Series] or None
            Spot prices per symbol.  If None, entry_price=1.0 is assumed.
        hmm_ranks : pd.Series or None
            Daily SPY HMM dominant-state rank (0/1/2), indexed by date.
            When supplied, ``step()`` receives the rank for that day and
            scales position size via ``regime_scale``.

        Returns
        -------
        pd.DataFrame
            Position log with one row per funding period.
        """
        all_idx = pd.DatetimeIndex([])
        for s in funding_series.values():
            all_idx = all_idx.union(s.index)
        all_idx = all_idx.sort_values()

        for ts in all_idx:
            rates = {sym: float(s.get(ts, 0.0)) for sym, s in funding_series.items()}
            px = (
                {sym: float(s.get(ts, 1.0)) for sym, s in prices.items()}
                if prices is not None
                else {sym: 1.0 for sym in self._symbols}
            )
            rank: int | None = None
            if hmm_ranks is not None:
                date_key = ts.date() if hasattr(ts, "date") else ts
                rank_val = hmm_ranks.get(date_key)
                if rank_val is not None and not pd.isna(rank_val):
                    rank = int(rank_val)
            self.step(rates, px, ts, hmm_rank=rank)

        self._save_outputs()
        return pd.DataFrame(self._position_log)

    # ── Live mode ────────────────────────────────────────────────────────────

    def run_live(self, poll_interval: float = 3600.0, initial_hmm_rank: int | None = None) -> None:
        """Poll Binance for funding rates and run indefinitely.

        Parameters
        ----------
        poll_interval : float
            Seconds between polls.  Defaults to 3600 (1 hour).
        initial_hmm_rank : int or None
            SPY HMM dominant-state rank at startup (0=bear, 1=neutral, 2=bull).
            Held constant until the process is restarted.  Refresh daily by
            restarting with a fresh rank from ``_compute_spy_hmm_ranks()``.
        """
        logger.info(
            "CarryStrategy live mode starting. symbols=%s poll=%.0fs regime_rank=%s",
            self._symbols, poll_interval, initial_hmm_rank,
        )
        current_rank = initial_hmm_rank
        polls_since_rank_refresh = 0
        # Refresh SPY rank every 24 polls (≈ once per day at 1h interval)
        rank_refresh_every = max(1, int(86_400 / poll_interval))

        while True:
            try:
                ts = pd.Timestamp.now(tz="UTC")
                rates: dict[str, float] = {}
                prices: dict[str, float] = {}
                for sym in self._symbols:
                    rates[sym] = _fetch_latest_funding_rate(sym)
                    prices[sym] = 1.0
                self.step(rates, prices, ts, hmm_rank=current_rank)
                self._save_outputs()
                polls_since_rank_refresh += 1
                if polls_since_rank_refresh >= rank_refresh_every:
                    logger.info("Daily SPY rank refresh due — restart with updated rank for accuracy.")
                    polls_since_rank_refresh = 0
            except Exception as exc:
                logger.error("step failed: %s", exc, exc_info=True)
            time.sleep(poll_interval)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_outputs(self) -> None:
        """Write position log and funding log to parquet."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        pos_path = self._output_dir / "positions.parquet"
        fund_path = self._output_dir / "funding_log.parquet"

        if self._position_log:
            pd.DataFrame(self._position_log).to_parquet(pos_path, index=False)
            logger.debug("Positions saved → %s (%d rows)", pos_path, len(self._position_log))

        fund_df = self._portfolio.funding_log_df()
        if not fund_df.empty:
            fund_df.to_parquet(fund_path, index=False)
            logger.debug("Funding log saved → %s (%d rows)", fund_path, len(fund_df))
