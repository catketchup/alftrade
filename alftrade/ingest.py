"""
ingest.py - Ingest real A-share data into the standard daily-bar panel.

Bridges external data sources into the schema the rest of alftrade expects —
a long-format table with columns ``date, symbol, open, high, low, close,
volume, amount, prev_close, limit_up_price, limit_down_price, is_suspended,
total_shares, market_return`` (the exact output shape of ``data.bars_to_frame``).

Real A-share data sources rarely provide every field.  This module fills the
gaps in two steps, both in polars:

  1. **Aggregation** — intraday bars (e.g. 1-minute parquet-by-date) are
     aggregated to daily OHLCV per symbol.
  2. **Derivation** — ``prev_close`` (previous trading day's close),
     ``limit_up_price`` / ``limit_down_price`` (board-specific ±limit rounded
     to the 0.01 tick — 10% main board, 20% STAR/ChiNext, 30% BSE),
     ``market_return`` (equal-weight cross-sectional mean return), and
     defaults for ``is_suspended`` (False) and ``total_shares`` (0.0).

Usage:
    from alftrade.ingest import ingest_parquet

    # 1-minute parquet-by-date (e.g. chn_equ_hf_1minbar layout)
    long_df = ingest_parquet("path/to/chn_equ_hf_1minbar", intraday=True)

    # → long_df is ready for compute_factor_panel() or frame_to_bars().
"""

from typing import Optional
import os
import re

import polars as pl

from alftrade.schema import BAR_COLUMNS


# Alias for backward compatibility: the canonical bar schema.
STANDARD_COLUMNS = BAR_COLUMNS


def limit_pct_for_symbol(symbol: str) -> float:
    """Return the daily price-limit fraction for a symbol's board.

    A-share boards have different daily price limits:

      * STAR market (科创板, ``688xxx``):        ±20%
      * ChiNext (创业板, ``300xxx`` / ``301xxx``): ±20%
      * Beijing Stock Exchange (北交所, ``8xxxxx`` / ``4xxxxx``): ±30%
      * Main board (all other 6-digit codes):      ±10%

    ST stocks (special treatment) are limited to ±5%, but the ST flag cannot
    be derived from the code alone, so it is not handled here.

    Args:
        symbol: A ticker such as ``"688072.SH"``, ``"300251.SZ"``, ``"830000.BJ"``
            or a bare 6-digit code. The 6-digit code is extracted, so an
            exchange prefix/suffix does not affect the result.
    """
    m = re.search(r"\d{6}", symbol)
    code = m.group(0) if m else ""
    if code.startswith(("688", "300", "301")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _limit_pct_expr(symbol_col: str = "symbol") -> pl.Expr:
    """Board-specific price-limit fraction as a polars column expression."""
    code = pl.col(symbol_col).str.extract(r"(\d{6})")
    return (
        pl.when(
            code.str.starts_with("688")
            | code.str.starts_with("300")
            | code.str.starts_with("301")
        )
        .then(0.20)
        .when(code.str.starts_with("8") | code.str.starts_with("4"))
        .then(0.30)
        .otherwise(0.10)
    )


def ingest_parquet(
    source_dir: str,
    symbol_col: str = "stock_code",
    time_col: str = "datetime",
    limit_pct: Optional[float] = None,
    intraday: bool = False,
) -> pl.DataFrame:
    """Aggregate + derive a standard daily-bar table from a parquet-by-date source.

    Args:
        source_dir: Directory of ``*.parquet`` files, one per trading date,
            each with one row per (symbol, bar).  Layout matches real
            A-share vendor feeds (e.g. ``chn_equ_hf_1minbar``).
        symbol_col: Column naming the ticker (default ``stock_code``).
        time_col: Timestamp column (default ``datetime``).  Used to extract the
            trading date, and — when ``intraday=True`` — to order bars.
        limit_pct: Daily price-limit fraction. If None (default), the
            **board-specific** limit is used (10% / 20% / 30% by code prefix —
            see ``limit_pct_for_symbol``). Pass a float to override uniformly.
        intraday: True if the source is intraday (1-minute) bars needing
            daily aggregation; False if it is already daily OHLCV (the
            default — real data is consumed at daily-bar granularity, matching
            the simulated data).

    Returns:
        A ``polars.DataFrame`` with ``STANDARD_COLUMNS``, sorted by
        ``(symbol, date)``, ready for ``factors.compute_factor_panel`` or
        ``data.frame_to_bars``.
    """
    lf = pl.scan_parquet(os.path.join(source_dir, "*.parquet"))

    if intraday:
        lf = lf.sort(time_col).with_columns(
            pl.col(time_col).dt.strftime("%Y%m%d").cast(pl.Int64).alias("date")
        )
        daily = (
            lf.group_by(["date", symbol_col])
            .agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
            ])
            .rename({symbol_col: "symbol"})
        )
    else:
        daily = (
            lf.with_columns(
                pl.col(time_col).dt.strftime("%Y%m%d").cast(pl.Int64).alias("date")
            )
            .rename({symbol_col: "symbol"})
            .select(["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
        )

    return _derive(daily, limit_pct)


def ingest_akshare(
    symbols: list[str],
    start_date: str,
    end_date: str,
    limit_pct: Optional[float] = None,
) -> pl.DataFrame:
    """Fetch daily bars from akshare and derive the standard schema.

    Requires network access. ``symbols`` are 6-digit codes without exchange
    suffix (e.g. "000001"); the exchange ("sh"/"sz"/"bj") is inferred from the
    leading digit. Uses the **Sina** source (``stock_zh_a_daily``), which is
    more stable than the Eastmoney source and returns English column names.

    Args:
        symbols: A-share tickers without exchange suffix (e.g. ["000001", "600000"]).
        start_date / end_date: "YYYYMMDD" range.
        limit_pct: Daily price-limit fraction. If None (default), the
            board-specific limit is used (10% / 20% / 30% by code prefix).

    Returns:
        A ``polars.DataFrame`` with ``STANDARD_COLUMNS``.
    """
    import akshare as ak
    import time

    frames = []
    for code in symbols:
        if code.startswith(("8", "4")):
            prefix, suffix = "bj", ".BJ"
        elif code.startswith(("6", "9")):
            prefix, suffix = "sh", ".SH"
        else:
            prefix, suffix = "sz", ".SZ"
        raw = ak.stock_zh_a_daily(
            symbol=prefix + code,
            start_date=start_date, end_date=end_date, adjust="",
        )
        # Sina columns (English): date, open, high, low, close, volume, amount, ...
        df = pl.from_pandas(raw).with_columns([
            pl.col("date").dt.strftime("%Y%m%d").cast(pl.Int64),
            pl.lit(code + suffix).alias("symbol"),
        ]).select(["date", "symbol", "open", "high", "low", "close", "volume", "amount"])

        frames.append(df)
        time.sleep(0.2)  # be polite to the data source

    long = pl.concat(frames)
    return _derive(long.lazy().sort(["symbol", "date"]), limit_pct)


# ── Derivation (shared) ────────────────────────────────────────────────

def _derive(daily: pl.LazyFrame, limit_pct: Optional[float] = None) -> pl.DataFrame:
    """Derive prev_close, board-specific limit prices, market_return, defaults.

    When ``limit_pct`` is None, each symbol gets its board-specific daily limit
    (10% main board, 20% STAR/ChiNext, 30% BSE) via ``_limit_pct_expr``.
    Passing a float applies that fraction uniformly (legacy behavior).
    """
    daily = daily.sort(["symbol", "date"])

    # prev_close = previous trading day's close (per symbol); fall back to
    # close on the very first day.
    daily = daily.with_columns(
        pl.col("close").shift(1).over("symbol").alias("prev_close")
    )
    daily = daily.with_columns(
        pl.when(pl.col("prev_close").is_null())
        .then(pl.col("close"))
        .otherwise(pl.col("prev_close"))
        .alias("prev_close")
    )

    # Limit prices = prev_close × (1 ± board_pct), rounded to the 0.01 tick.
    pct = _limit_pct_expr() if limit_pct is None else pl.lit(limit_pct)
    daily = daily.with_columns([
        (pl.col("prev_close") * (1.0 + pct)).round(2).alias("limit_up_price"),
        (pl.col("prev_close") * (1.0 - pct)).round(2).alias("limit_down_price"),
        pl.lit(False).alias("is_suspended"),
        pl.lit(0.0).alias("total_shares"),
    ])

    # market_return = equal-weight cross-sectional mean daily return.
    daily = daily.with_columns(
        ((pl.col("close") - pl.col("prev_close")) / pl.col("prev_close")).alias("_ret")
    )
    mkt = daily.group_by("date").agg(pl.col("_ret").mean().alias("market_return"))
    daily = daily.join(mkt, on="date", how="left").drop("_ret")

    return daily.collect().select(STANDARD_COLUMNS)
