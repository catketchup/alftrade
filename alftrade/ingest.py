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
     ``limit_up_price`` / ``limit_down_price`` (±``limit_pct`` rounded to the
     0.01 tick), ``market_return`` (equal-weight cross-sectional mean return),
     and defaults for ``is_suspended`` (False) and ``total_shares`` (0.0).

Usage:
    from alftrade.ingest import ingest_parquet

    # 1-minute parquet-by-date (e.g. chn_equ_hf_1minbar layout)
    long_df = ingest_parquet("path/to/chn_equ_hf_1minbar", intraday=True)

    # → long_df is ready for compute_factor_panel() or frame_to_bars().
"""

from typing import Optional
import os

import polars as pl

from alftrade.schema import BAR_COLUMNS


# Alias for backward compatibility: the canonical bar schema.
STANDARD_COLUMNS = BAR_COLUMNS


def ingest_parquet(
    source_dir: str,
    symbol_col: str = "stock_code",
    time_col: str = "datetime",
    limit_pct: float = 0.10,
    intraday: bool = True,
) -> pl.DataFrame:
    """Aggregate + derive a standard daily-bar table from a parquet-by-date source.

    Args:
        source_dir: Directory of ``*.parquet`` files, one per trading date,
            each with one row per (symbol, bar).  Layout matches real
            A-share vendor feeds (e.g. ``chn_equ_hf_1minbar``).
        symbol_col: Column naming the ticker (default ``stock_code``).
        time_col: Timestamp column (default ``datetime``).  Used to extract the
            trading date, and — when ``intraday=True`` — to order bars.
        limit_pct: Daily price-limit fraction for limit-up/down derivation
            (default 0.10 = ±10%).
        intraday: True if the source is intraday (1-minute) bars needing
            daily aggregation; False if it is already daily OHLCV.

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
    limit_pct: float = 0.10,
) -> pl.DataFrame:
    """Fetch daily bars from akshare and derive the standard schema.

    Requires network access. ``symbols`` are 6-digit codes (e.g. "000001");
    the exchange suffix (".SZ" / ".SH") is inferred from the leading digit.

    Args:
        symbols: A-share tickers without exchange suffix (e.g. ["000001", "600000"]).
        start_date / end_date: "YYYYMMDD" range.
        limit_pct: Daily price-limit fraction.

    Returns:
        A ``polars.DataFrame`` with ``STANDARD_COLUMNS``.
    """
    import akshare as ak

    frames = []
    for code in symbols:
        suffix = ".SH" if code.startswith(("6", "9")) else ".SZ"
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date, adjust="",
        )
        # akshare columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, ...
        df = pl.from_pandas(raw).rename({
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
        }).with_columns([
            pl.col("date").cast(pl.Utf8).str.replace_all("-", "").cast(pl.Int64),
            pl.lit(code + suffix).alias("symbol"),
        ]).select(["date", "symbol", "open", "high", "low", "close", "volume", "amount"])

        frames.append(df)

    long = pl.concat(frames)
    return _derive(long.lazy().sort(["symbol", "date"]), limit_pct)


# ── Derivation (shared) ────────────────────────────────────────────────

def _derive(daily: pl.LazyFrame, limit_pct: float) -> pl.DataFrame:
    """Derive prev_close, limit prices, market_return, and defaults."""
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

    # Limit prices = prev_close × (1 ± limit_pct), rounded to the 0.01 tick.
    daily = daily.with_columns([
        (pl.col("prev_close") * (1.0 + limit_pct)).round(2).alias("limit_up_price"),
        (pl.col("prev_close") * (1.0 - limit_pct)).round(2).alias("limit_down_price"),
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
