"""
store.py - Parquet-by-date data layer (simulated A-share market data I/O).

Mirrors the real A-share data-vendor layout used elsewhere in this workspace
(one parquet file per trading date, e.g. ``chn_equ_hf_*``).  The purpose is to
exercise the *real* data-access path — polars reads from disk, one date at a
time — without having actual market data.

Layout written by ``write_market_parquet``::

    root/
      marketdata/            # one file per date, one row per (symbol, date)
        20200101.parquet
        20200102.parquet
        ...
      index.parquet          # date + market_return (index return, common to all)

The reader side is split in two:

  * ``ParquetPanel`` — a ``collections.abc.Mapping`` that behaves like the
    in-memory ``bars_by_date`` dict but lazily reads one parquet file per
    ``__getitem__`` call (with a small LRU cache).  This is a drop-in
    replacement for the ``dict[date, dict[symbol, DailyBar]]`` the engine,
    calibrator, and portfolio consume.

  * ``load_market_with_factors`` — reads the whole panel, joins the index,
    computes the 11 factors with polars, and returns the factor dict that the
    ``TradingEngine`` consumes.
"""

from collections import OrderedDict
from collections.abc import Mapping
import glob
import os
from typing import Optional

import polars as pl

from alftrade.data import DailyBar, StockUniverse
from alftrade.factors import FACTOR_NAMES, compute_factor_panel, factor_panel_to_dict
from alftrade.schema import MARKETDATA_COLUMNS


# ── Writing ─────────────────────────────────────────────────────────────

def write_market_parquet(
    bars_by_date: dict[int, dict[str, DailyBar]],
    universe: StockUniverse,
    market_returns: dict[int, float],
    root: str,
) -> None:
    """Write an in-memory panel to a parquet-by-date directory tree.

    Args:
        bars_by_date: ``{date: {symbol: DailyBar}}`` (e.g. from
            ``generate_simulated_market``).
        universe: The stock universe (used only for metadata; the marketdata
            files carry everything needed to reconstruct bars).
        market_returns: ``{date: float}`` daily index return.
        root: Output directory. Creates ``root/marketdata`` and ``root/index.parquet``.
    """
    marketdata_dir = os.path.join(root, "marketdata")
    os.makedirs(marketdata_dir, exist_ok=True)

    for date, day in bars_by_date.items():
        rows = [
            {c: getattr(b, c) for c in MARKETDATA_COLUMNS}
            for b in day.values()
        ]
        pl.DataFrame(rows).write_parquet(os.path.join(marketdata_dir, f"{date}.parquet"))

    idx = pl.DataFrame({
        "date": list(market_returns.keys()),
        "market_return": [float(market_returns[d]) for d in market_returns.keys()],
    })
    idx.write_parquet(os.path.join(root, "index.parquet"))


# ── Reading ─────────────────────────────────────────────────────────────

class ParquetPanel(Mapping):
    """Lazy, cached ``{date: {symbol: DailyBar}}`` over a parquet-by-date directory.

    Behaves like the in-memory ``bars_by_date`` dict used throughout the
    framework, but each ``panel[date]`` triggers a single
    ``pl.read_parquet`` (cached in a bounded LRU).  Inheriting ``Mapping``
    provides ``.get()``, ``__contains__``, ``.keys()``, ``.items()`` for free,
    so it works wherever a plain dict of daily bars did.

    Args:
        marketdata_dir: Directory containing ``{date}.parquet`` files.
        market_returns: ``{date: float}``, injected into each reconstructed bar.
        cache_size: Maximum number of dates kept in the LRU cache.
    """

    def __init__(
        self,
        marketdata_dir: str,
        market_returns: dict[int, float],
        cache_size: int = 64,
    ):
        self._dir = marketdata_dir
        self._market_returns = market_returns
        self._cache: OrderedDict[int, dict[str, DailyBar]] = OrderedDict()
        self._cache_size = cache_size
        self._dates: list[int] = sorted(
            int(os.path.splitext(os.path.basename(p))[0])
            for p in glob.glob(os.path.join(marketdata_dir, "*.parquet"))
        )

    def __getitem__(self, date: int) -> dict[str, DailyBar]:
        if date not in self._cache:
            bars = self._read_date(date)
            self._cache[date] = bars
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # evict oldest (LRU)
        return self._cache[date]

    def __iter__(self):
        return iter(self._dates)

    def __len__(self) -> int:
        return len(self._dates)

    @property
    def dates(self) -> list[int]:
        """All dates present in the directory, ascending."""
        return list(self._dates)

    def _read_date(self, date: int) -> dict[str, DailyBar]:
        path = os.path.join(self._dir, f"{date}.parquet")
        if not os.path.exists(path):
            raise KeyError(date)
        mkt_ret = self._market_returns.get(date, 0.0)
        df = pl.read_parquet(path)
        bars: dict[str, DailyBar] = {}
        for row in df.iter_rows(named=True):
            bars[row["symbol"]] = DailyBar(
                date=int(row["date"]),
                symbol=row["symbol"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                amount=float(row["amount"]),
                prev_close=float(row["prev_close"]),
                limit_up_price=float(row["limit_up_price"]),
                limit_down_price=float(row["limit_down_price"]),
                is_suspended=bool(row["is_suspended"]),
                total_shares=float(row["total_shares"]),
                market_return=float(mkt_ret),
            )
        return bars


def read_market_frame(root: str) -> pl.DataFrame:
    """Read all marketdata + index into a single long table (canonical schema).

    Joins the per-date marketdata parquet with the index on ``date``, yielding
    one row per (symbol, date) with every ``schema.BAR_COLUMNS``.
    """
    market = pl.scan_parquet(os.path.join(root, "marketdata", "*.parquet"))
    index = pl.scan_parquet(os.path.join(root, "index.parquet"))
    return market.join(index, on="date", how="left").collect()


def load_market_parquet(root: str) -> tuple[ParquetPanel, StockUniverse, dict[int, float]]:
    """Load a parquet-by-date directory into (panel, universe, market_returns).

    The return signature matches ``generate_simulated_market``, so this is a
    drop-in replacement for the in-memory generator.
    """
    idx = pl.read_parquet(os.path.join(root, "index.parquet"))
    market_returns = {int(r["date"]): float(r["market_return"]) for r in idx.iter_rows(named=True)}

    panel = ParquetPanel(os.path.join(root, "marketdata"), market_returns)

    # Rebuild the universe with a projection-only scan (polars reads only the
    # date + symbol columns, thanks to parquet column pruning).
    df = pl.scan_parquet(os.path.join(root, "marketdata", "*.parquet")) \
           .select(["date", "symbol"]).collect()
    universe = StockUniverse()
    for (date,), grp in df.group_by("date", maintain_order=True):
        universe.add(int(date), grp["symbol"].to_list())

    return panel, universe, market_returns


def load_factor_panel(root: str) -> dict[int, dict[str, dict[str, float]]]:
    """Read the parquet panel and compute all factors via polars.

    Returns ``{date: {symbol: {factor: value}}}`` ready for the TradingEngine.
    """
    long_df = read_market_frame(root)
    factor_df = compute_factor_panel(long_df)
    return factor_panel_to_dict(factor_df)


def load_market_with_factors(
    root: str,
) -> tuple[ParquetPanel, dict[int, dict[str, dict[str, float]]], StockUniverse, dict[int, float]]:
    """Load both the bar panel and the precomputed factors in one call.

    Returns ``(panel, factors, universe, market_returns)``.
    """
    panel, universe, market_returns = load_market_parquet(root)
    factors = load_factor_panel(root)
    return panel, factors, universe, market_returns
