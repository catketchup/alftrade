"""
factors.py - Cross-sectional daily factor computation (polars-vectorized).

This module computes all 11 daily factors in a single vectorized pass over a
long-format ``polars.DataFrame`` — one row per (symbol, date).  This replaces
the earlier per-stock ``deque`` rolling calculator with a batch, group-by
implementation that is idiomatic polars and scales to real A-share universes.

Design notes:
  * Every factor is expressed as a ``pl.Expr`` with ``.over("symbol")`` so that
    rolling / shift / ewm windows are computed within each stock's time series.
  * Division is guarded with ``when(den.abs() > eps).then(num/den).otherwise(fb)``
    because polars yields ``inf`` for ``x/0`` and ``NaN`` for ``0/0`` (verified).
  * Warm-up NaNs are filled: ``0.0`` for every factor except ``market_beta``,
    which falls back to ``1.0`` (neutral beta) — matching the previous
    calculator's "0 until enough history" convention.

Factor list (11, all computable from OHLCV + prev_close + market_return):
    daily_return      = (close - prev_close) / prev_close
    overnight_return  = (open  - prev_close) / prev_close
    intraday_return   = (close - open)       / open
    momentum_5d       = (close - close[t-5])  / close[t-5]
    momentum_20d      = (close - close[t-20]) / close[t-20]
    ma_dev_20         = (close - SMA20)       / SMA20
    volatility_20d    = std(daily_return, 20)  (population, ddof=0)
    volume_ratio      = volume / SMA20(volume)
    rsi_14            = Wilder RSI(14), rescaled to [-1, 1]
    atr_pct_14        = Wilder ATR(14) / close
    market_beta       = rolling OLS beta of daily_return on market_return (60d)

Usage:
    df = pl.DataFrame(...)                 # long table, one row per (symbol, date)
    factors = compute_factor_panel(df)     # [date, symbol, + 11 factor columns]
    by_date = factor_panel_to_dict(factors) # {date: {symbol: {factor: value}}}
"""

import polars as pl


# Ordered factor names — the canonical ordering used everywhere in the system
# (calibration, profiler, engine). Code that iterates factors should use this
# to guarantee consistent column ordering between training and inference.
FACTOR_NAMES: list[str] = [
    "daily_return",
    "overnight_return",
    "intraday_return",
    "momentum_5d",
    "momentum_20d",
    "ma_dev_20",
    "volatility_20d",
    "volume_ratio",
    "rsi_14",
    "atr_pct_14",
    "market_beta",
]


def _sdiv(num: pl.Expr, den: pl.Expr, fallback: float = 0.0) -> pl.Expr:
    """Safe division: returns ``fallback`` when the divisor is ~0 or null."""
    return pl.when(den.abs() > 1e-12).then(num / den).otherwise(fallback)


def compute_factor_panel(
    df: pl.DataFrame,
    sma_window: int = 20,
    vol_window: int = 20,
    rsi_window: int = 14,
    atr_window: int = 14,
    beta_window: int = 60,
    beta_min_samples: int = 12,
) -> pl.DataFrame:
    """Compute all factors for a long-format panel in one vectorized pass.

    Args:
        df: Long table with columns ``date, symbol, open, high, low, close,
            volume, prev_close, market_return`` (extra columns are ignored).
        sma_window: SMA lookback (default 20).
        vol_window: Volatility lookback (default 20).
        rsi_window: RSI lookback (default 14).
        atr_window: ATR lookback (default 14).
        beta_window: Market-beta rolling window (default 60).
        beta_min_samples: Minimum points for the beta regression (default 12).

    Returns:
        A ``polars.DataFrame`` with columns ``[date, symbol] + FACTOR_NAMES``,
        sorted by ``(symbol, date)``, with warm-up NaNs filled (0.0, or 1.0
        for market_beta).
    """
    df = df.sort(["symbol", "date"])

    # Materialize the per-symbol close diff once: RSI needs a *second* window
    # (ewm) on top of the diff, and polars forbids nesting window expressions
    # inside a window context.  By materialising `_delta` as a plain column
    # first, the RSI ewm applies a single `.over("symbol")` to a plain column.
    df = df.with_columns(
        pl.col("close").diff().over("symbol").alias("_delta")
    )

    # ── Price-based factors ──
    daily_return = _sdiv(pl.col("close") - pl.col("prev_close"), pl.col("prev_close"))
    overnight_return = _sdiv(pl.col("open") - pl.col("prev_close"), pl.col("prev_close"))
    intraday_return = _sdiv(pl.col("close") - pl.col("open"), pl.col("open"))

    # Momentum via shifted close within each symbol.
    momentum_5d = _sdiv(
        pl.col("close") - pl.col("close").shift(5).over("symbol"),
        pl.col("close").shift(5).over("symbol"),
    )
    momentum_20d = _sdiv(
        pl.col("close") - pl.col("close").shift(20).over("symbol"),
        pl.col("close").shift(20).over("symbol"),
    )

    # Deviation from the 20-day SMA.
    sma20 = pl.col("close").rolling_mean(sma_window, min_samples=1).over("symbol")
    ma_dev_20 = _sdiv(pl.col("close") - sma20, sma20)

    # ── Volatility (population std of daily returns, ddof=0) ──
    volatility_20d = daily_return.rolling_std(
        vol_window, ddof=0, min_samples=2
    ).over("symbol")

    # ── Volume ratio ──
    vol_sma = pl.col("volume").rolling_mean(sma_window, min_samples=1).over("symbol")
    volume_ratio = _sdiv(pl.col("volume"), vol_sma, fallback=1.0)

    # ── RSI via Wilder smoothing (EMA with alpha = 1/n), rescaled to [-1, 1] ──
    # RSI = 100 * avg_gain / (avg_gain + avg_loss): pure up-trend → 100,
    # pure down-trend → 0, warm-up (both ≈ 0) → 50 (neutral).
    gain = pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0)
    loss = pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / rsi_window, adjust=False, min_samples=1).over("symbol")
    avg_loss = loss.ewm_mean(alpha=1.0 / rsi_window, adjust=False, min_samples=1).over("symbol")
    sum_gain_loss = avg_gain + avg_loss
    rsi_raw = pl.when(sum_gain_loss > 1e-12).then(
        100.0 * avg_gain / sum_gain_loss
    ).otherwise(50.0)
    rsi_14 = (rsi_raw - 50.0) / 50.0

    # ── ATR via Wilder smoothing of True Range, as a fraction of close ──
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("prev_close")).abs(),
        (pl.col("low") - pl.col("prev_close")).abs(),
    )
    atr = tr.ewm_mean(alpha=1.0 / atr_window, adjust=False, min_samples=1).over("symbol")
    atr_pct_14 = _sdiv(atr, pl.col("close"))

    # ── Market beta: rolling OLS beta via the covariance identity ──
    # cov(x, y) = E[xy] - E[x]E[y];  var(y) = E[y^2] - E[y]^2   (population)
    m = pl.col("market_return")
    d = daily_return
    dm = d * m
    cov = (
        dm.rolling_mean(beta_window, min_samples=beta_min_samples).over("symbol")
        - d.rolling_mean(beta_window, min_samples=beta_min_samples).over("symbol")
        * m.rolling_mean(beta_window, min_samples=beta_min_samples).over("symbol")
    )
    var = (
        (m * m).rolling_mean(beta_window, min_samples=beta_min_samples).over("symbol")
        - m.rolling_mean(beta_window, min_samples=beta_min_samples).over("symbol") ** 2
    )
    market_beta = _sdiv(cov, var, fallback=1.0)

    out = df.with_columns([
        daily_return.fill_null(0.0).alias("daily_return"),
        overnight_return.fill_null(0.0).alias("overnight_return"),
        intraday_return.fill_null(0.0).alias("intraday_return"),
        momentum_5d.fill_null(0.0).alias("momentum_5d"),
        momentum_20d.fill_null(0.0).alias("momentum_20d"),
        ma_dev_20.fill_null(0.0).alias("ma_dev_20"),
        volatility_20d.fill_null(0.0).alias("volatility_20d"),
        volume_ratio.fill_null(1.0).alias("volume_ratio"),
        rsi_14.fill_null(0.0).alias("rsi_14"),
        atr_pct_14.fill_null(0.0).alias("atr_pct_14"),
        market_beta.fill_null(1.0).alias("market_beta"),
    ])
    return out.select(["date", "symbol"] + FACTOR_NAMES)


def factor_panel_to_dict(
    factor_df: pl.DataFrame,
) -> dict[int, dict[str, dict[str, float]]]:
    """Convert a factor panel into ``{date: {symbol: {factor: value}}}``.

    This is the shape the ``TradingEngine`` consumes, so the factor panel can
    be precomputed once and looked up per date during the backtest.

    Args:
        factor_df: Output of ``compute_factor_panel`` (or an equivalent table
            with columns ``date, symbol`` + ``FACTOR_NAMES``).

    Returns:
        Nested dict keyed by date then symbol then factor name.
    """
    result: dict[int, dict[str, dict[str, float]]] = {}
    for (date,), grp in factor_df.group_by("date", maintain_order=True):
        day: dict[str, dict[str, float]] = {}
        for row in grp.iter_rows(named=True):
            day[row["symbol"]] = {n: float(row[n]) for n in FACTOR_NAMES}
        result[int(date)] = day
    return result
