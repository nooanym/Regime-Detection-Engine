"""Append-only trade log with parquet / CSV serialisation (Phase 34)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from rde.trading.paper_portfolio import Fill

logger = logging.getLogger(__name__)

# Column names used when (de)serialising fills.
_FILL_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "price",
    "slippage",
    "commission",
]


class TradeLog:
    """Append-only log of :class:`~rde.trading.paper_portfolio.Fill` objects.

    Supports conversion to / from :class:`pandas.DataFrame` and
    serialisation to parquet and CSV.
    """

    def __init__(self) -> None:
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, fill: Fill) -> None:
        """Append *fill* to the log.

        Args:
            fill: A completed fill record.
        """
        self._fills.append(fill)
        logger.debug("TradeLog: appended fill %s %s %.6f", fill.side, fill.symbol, fill.quantity)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the log to a :class:`pandas.DataFrame`.

        Returns:
            DataFrame with columns ``timestamp``, ``symbol``, ``side``,
            ``quantity``, ``price``, ``slippage``, ``commission``.
            Empty DataFrame (with correct columns) if no fills have been logged.
        """
        if not self._fills:
            return pd.DataFrame(columns=_FILL_COLUMNS)
        rows = [
            {
                "timestamp": f.timestamp,
                "symbol": f.symbol,
                "side": f.side,
                "quantity": f.quantity,
                "price": f.price,
                "slippage": f.slippage,
                "commission": f.commission,
            }
            for f in self._fills
        ]
        return pd.DataFrame(rows, columns=_FILL_COLUMNS)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_parquet(self, path: Path) -> None:
        """Write fills to a parquet file at *path*.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        logger.info("TradeLog: saved %d fills to %s", len(self), path)

    def save_csv(self, path: Path) -> None:
        """Write fills to a CSV file at *path*.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("TradeLog: saved %d fills to %s", len(self), path)

    @classmethod
    def load_parquet(cls, path: Path) -> TradeLog:
        """Load a :class:`TradeLog` from a parquet file.

        Args:
            path: Source parquet file previously written by
                :meth:`save_parquet`.

        Returns:
            A new :class:`TradeLog` populated with the fills from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        df = pd.read_parquet(path)
        log = cls()
        for _, row in df.iterrows():
            fill = Fill(
                timestamp=pd.Timestamp(row["timestamp"]),
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                slippage=float(row["slippage"]),
                commission=float(row["commission"]),
            )
            log._fills.append(fill)
        logger.info("TradeLog: loaded %d fills from %s", len(log), path)
        return log

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._fills)

    def __iter__(self) -> Iterator[Fill]:
        return iter(self._fills)
