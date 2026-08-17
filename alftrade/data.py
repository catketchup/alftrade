"""
data.py - Daily bar data model and simulated A-share market generator.

This module is the daily-frequency analogue of ``hftrade.engine.Tick`` and
``generate_simulated_ticks``. The key differences from the high-frequency
version are driven by the A-share daily cross-sectional setting:

  * A single observation is a **daily OHLCV bar** per stock, not a tick.
  * The simulator produces a **panel** of bars (N stocks × T days) rather
    than a single tick stream, because cross-sectional strategies need a
    universe of comparable instruments per date.
  * Each bar carries A-share-specific metadata needed by the risk layer:
    ``prev_close`` (for daily-return and price-limit computation),
    ``limit_up_price`` / ``limit_down_price`` (the ±10% daily price band),
    and an ``is_suspended`` flag (A-share stocks are frequently halted).

Simulator design — why prices are generated with a factor structure
-------------------------------------------------------------------
A purely-random walk would make every factor's predictive IC ≈ 0, which is
useless for testing the calibrator and profiler. Instead the simulator
generates next-day returns as a known linear function of *past* factors:

    r_{i,t} = α_i + Σ_k β_k · factor_{i,k,t-1} + ε_{i,t}

The factors on the right-hand side are computed from prices *up to t-1*, so
there is no look-ahead. This creates genuine, controllable predictability:
the calibrator should recover β_k ≈ the true loadings, and the profiler
should report high IC / monotonicity for factors with large |β_k|.

The default loadings encode a small-cap / low-volatility / momentum tilt,
which is a reasonable caricature of A-share daily cross-sectional structure.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from alftrade.schema import BAR_COLUMNS, validate_bar_frame


@dataclass
class DailyBar:
    """A single stock's daily OHLCV bar with A-share risk metadata.

    Attributes:
        date: Trading-day ordinal (integer; days are consecutive across the
              simulation/calendar). Comparable with ``==`` / ``<``.
        symbol: Ticker, e.g. "000001.SZ".
        open, high, low, close: Daily prices (yuan).
        volume: Traded volume (shares).
        amount: Traded turnover (yuan) = average price × volume.
        prev_close: Previous trading day's close. Needed for:
            - daily_return = (close - prev_close) / prev_close
            - price-limit band computation (±limit_pct around prev_close).
        limit_up_price: Upper daily price limit = round(prev_close × (1+p)).
        limit_down_price: Lower daily price limit = round(prev_close × (1-p)).
        is_suspended: True if the stock is halted on this date (no trading
                      allowed; risk layer blocks both buys and sells).
        total_shares: Shares outstanding (for a true turnover ratio). If 0,
                      the factor calculator falls back to an amount-based
                      turnover proxy.
    """
    date: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    prev_close: float
    limit_up_price: float = 0.0
    limit_down_price: float = 0.0
    is_suspended: bool = False
    total_shares: float = 0.0
    market_return: float = 0.0

    @property
    def mid_price(self) -> float:
        """Convenience: a representative execution price for the bar.

        We use the close as the backtest execution price (see engine.py for
        the same-bar execution convention). Mirrors hftrade's mid_price use
        as the trade reference price.
        """
        return self.close

    @property
    def is_limit_up(self) -> bool:
        """True if the close is at/above the limit-up price (cannot buy)."""
        return self.limit_up_price > 0 and self.close >= self.limit_up_price - 1e-9

    @property
    def is_limit_down(self) -> bool:
        """True if the close is at/below the limit-down price (cannot sell)."""
        return self.limit_down_price > 0 and self.close <= self.limit_down_price + 1e-9


def round_price_to_tick(price: float, tick: float = 0.01) -> float:
    """Round a price to the A-share minimum tick (1 fen = 0.01 yuan).

    Price-limit bands and order prices must be on the tick grid.
    """
    return round(price / tick) * tick


def compute_limit_prices(prev_close: float, limit_pct: float) -> tuple[float, float]:
    """Compute (limit_up, limit_down) prices for an A-share daily bar.

    A-share daily price limits:
        Main board / ChiNext: ±10% (limit_pct=0.10)
        STAR board / BSE:     ±20% (limit_pct=0.20)
        ST stocks:            ±5%  (limit_pct=0.05)

    Prices are rounded to the 0.01 yuan tick, matching exchange convention.
    """
    if prev_close <= 0:
        return 0.0, 0.0
    up = round_price_to_tick(prev_close * (1.0 + limit_pct))
    down = round_price_to_tick(prev_close * (1.0 - limit_pct))
    return up, down


class StockUniverse:
    """Tracks the set of tradeable symbols on each trading date.

    A-share universes change over time due to IPOs, delistings, and
    suspensions. The calibrator and profiler use this to know which
    (date, symbol) pairs are valid cross-sectional members.

    Attributes:
        symbols: All symbols ever seen, in deterministic order.
    """

    def __init__(self):
        self._active: dict[int, set[str]] = {}
        self._all: list[str] = []

    def add(self, date: int, symbols) -> None:
        """Register the set of active symbols on a given date.

        Membership is stored as a set (for O(1) ``active_on`` lookups), but
        first-appearance order is preserved by iterating the input sequence
        directly rather than the set.
        """
        sym_list = list(symbols)
        self._active[date] = set(sym_list)
        for s in sym_list:
            if s not in self._all:
                self._all.append(s)

    def active_on(self, date: int) -> set[str]:
        """Return the set of active symbols on `date` (empty if unknown)."""
        return self._active.get(date, set())

    @property
    def symbols(self) -> list[str]:
        """All symbols ever seen, in first-appearance order."""
        return list(self._all)

    @property
    def dates(self) -> list[int]:
        """All registered dates in ascending order."""
        return sorted(self._active.keys())


# ── Simulated market generator ─────────────────────────────────────────


# Default "true" factor loadings used by the simulator to drive next-day
# returns. These mirror the kinds of effects documented in the A-share daily
# cross-sectional literature, scaled so that the predicted daily return has
# std ≈ 0.3% — well inside the ±10% daily price-limit band, so prices stay
# realistic and the clip rarely binds. With idiosyncratic noise std ≈ 1.4%,
# this gives per-factor IC ≈ 0.15–0.25 (strong but not absurd; real A-share
# daily factors are typically IC ≈ 0.03–0.08).
#   * negative 5-day momentum (short-term reversal is strong in A-shares)
#   * negative volatility (low-vol anomaly)
#   * positive turnover (active names keep running short-term)
#   * positive ma_dev (mild trend continuation vs the 20d average)
DEFAULT_FACTOR_LOADINGS: dict[str, float] = {
    "momentum_5d": -0.0010,     # 5-day reversal
    "volatility_20d": -0.0025,  # low-vol anomaly (strongest driver)
    "volume_ratio": 0.0006,     # turnover / liquidity tilt
    "ma_dev_20": 0.0008,        # mild trend continuation
}


def generate_simulated_market(
    n_stocks: int = 30,
    n_days: int = 300,
    seed: int = 42,
    start_date: int = 20200101,
    factor_loadings: Optional[dict[str, float]] = None,
    limit_pct: float = 0.10,
    burn_in: int = 30,
) -> tuple[dict[int, dict[str, DailyBar]], StockUniverse]:
    """Generate a simulated A-share daily panel with controllable factor signal.

    The generator produces ``n_stocks`` symbols over ``n_days`` trading days.
    Next-day returns are a linear function of *past* factors (see module
    docstring), so factors have genuine predictive power for forward returns.

    Args:
        n_stocks: Number of symbols in the universe.
        n_days: Number of trading days to generate.
        seed: RNG seed for reproducibility.
        start_date: Integer label of the first trading day.
        factor_loadings: True β used to drive returns. Defaults to
            ``DEFAULT_FACTOR_LOADINGS``. Set a loading to 0 to remove that
            factor's predictive power.
        limit_pct: Daily price-limit fraction (default 0.10 = ±10%).
        burn_in: Number of leading days generated as pure noise so that
            rolling factors (20d SMA / volatility) are well-defined before
            the factor-driven regime begins.

    Returns:
        (bars_by_date, universe, market_returns) where
            bars_by_date[date] = {symbol: DailyBar}
            universe.active_on(date) = set of symbols with a bar on that date.
            market_returns[date] = daily market index return (common to all stocks).
    """
    if n_stocks < 2:
        raise ValueError("Need at least 2 stocks for a cross-sectional market.")
    if n_days < burn_in + 10:
        raise ValueError(f"n_days={n_days} too small for burn_in={burn_in}.")

    loadings = dict(DEFAULT_FACTOR_LOADINGS)
    if factor_loadings is not None:
        loadings.update(factor_loadings)

    rng = np.random.default_rng(seed)

    # Symbol names styled like A-share tickers (Shenzhen 6-digit form).
    symbols = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    # Per-stock latent drift and base scale — gives cross-sectional spread.
    alpha = rng.normal(0.0, 0.0005, size=n_stocks)        # daily drift
    base_price = rng.uniform(15.0, 60.0, size=n_stocks)   # starting price
    base_volume = rng.uniform(5e5, 3e6, size=n_stocks)    # shares/day
    total_shares = rng.uniform(5e8, 3e9, size=n_stocks)   # shares outstanding
    vol_noise_scale = rng.uniform(0.008, 0.020, size=n_stocks)  # idio vol

    # ── Market (index) return & per-stock CAPM beta ──
    # A hidden common factor drives all stocks to varying degrees, producing
    # realistic cross-sectional correlation and creating a genuine market-beta
    # factor that the calibrator can recover.
    market_returns_arr: np.ndarray = rng.normal(0.0002, 0.005, size=n_days)
    stock_market_betas: np.ndarray = rng.uniform(0.5, 1.5, size=n_stocks)
    market_returns: dict[int, float] = {}

    # Price and volume history matrices: shape (T, N).
    close = np.zeros((n_days, n_stocks))
    openp = np.zeros((n_days, n_stocks))
    high = np.zeros((n_days, n_stocks))
    low = np.zeros((n_days, n_stocks))
    volume = np.zeros((n_days, n_stocks))
    amount = np.zeros((n_days, n_stocks))

    close[0] = base_price
    openp[0] = base_price * (1.0 + rng.normal(0, 0.005, n_stocks))
    high[0] = np.maximum(openp[0], close[0]) * (1.0 + np.abs(rng.normal(0, 0.004, n_stocks)))
    low[0] = np.minimum(openp[0], close[0]) * (1.0 - np.abs(rng.normal(0, 0.004, n_stocks)))
    volume[0] = base_volume * rng.lognormal(0, 0.3, n_stocks)

    # The factor-driven regime needs at least 21 bars of history for the
    # deepest lookback (20-day momentum). Effective burn-in covers both.
    effective_burn_in = max(burn_in, 21)

    bars_by_date: dict[int, dict[str, DailyBar]] = {}
    universe = StockUniverse()

    for t in range(n_days):
        prev_close = close[t - 1] if t > 0 else base_price
        mkt_ret = market_returns_arr[t]

        if t < effective_burn_in:
            # Pure-noise regime so rolling windows fill up.
            ret = alpha + stock_market_betas * mkt_ret + rng.normal(0, 1, n_stocks) * vol_noise_scale
        else:
            # Factor-driven regime: use factors computed from prices[:t]
            # (i.e. data up to and including day t-1 — no look-ahead).
            factor_mat = _sim_factor_matrix(close, volume, t, list(loadings.keys()))
            # Cross-sectional standardization (matches calibration transform).
            mu = factor_mat.mean(axis=0)
            sd = factor_mat.std(axis=0)
            sd_safe = np.where(sd > 1e-12, sd, 1.0)
            z = (factor_mat - mu) / sd_safe
            betas = np.array([loadings[n] for n in loadings.keys()])
            pred = z @ betas
            ret = alpha + pred + stock_market_betas * mkt_ret + rng.normal(0, 1, n_stocks) * vol_noise_scale

        # Clip returns to the daily price-limit band so prices stay tradeable.
        ret = np.clip(ret, -limit_pct * 0.995, limit_pct * 0.995)
        close[t] = prev_close * (1.0 + ret)

        # Build realistic OHLC around the close-to-close return.
        overnight = rng.normal(0, 0.004, n_stocks)
        openp[t] = prev_close * (1.0 + overnight)
        intraday_range = np.abs(rng.normal(0, 0.005, n_stocks)) + 0.001
        high[t] = np.maximum(openp[t], close[t]) * (1.0 + intraday_range)
        low[t] = np.minimum(openp[t], close[t]) * (1.0 - intraday_range)
        # Volume scales with |return| (active days trade more).
        vol = base_volume * (1.0 + 4.0 * np.abs(ret)) * rng.lognormal(0, 0.3, n_stocks)
        volume[t] = vol

        amount[t] = vol * (openp[t] + high[t] + low[t] + close[t]) / 4.0

        # Construct DailyBar objects and feed them into each calculator so
        # that the next iteration's factor computation sees today's data.
        date = start_date + t
        day_bars: dict[str, DailyBar] = {}
        for i, sym in enumerate(symbols):
            lim_up, lim_dn = compute_limit_prices(prev_close[i], limit_pct)
            bar = DailyBar(
                date=date,
                symbol=sym,
                open=float(openp[t, i]),
                high=float(high[t, i]),
                low=float(low[t, i]),
                close=float(close[t, i]),
                volume=float(volume[t, i]),
                amount=float(amount[t, i]),
                prev_close=float(prev_close[i]),
                limit_up_price=float(lim_up),
                limit_down_price=float(lim_dn),
                is_suspended=False,
                total_shares=float(total_shares[i]),
                market_return=float(mkt_ret),
            )
            day_bars[sym] = bar

        bars_by_date[date] = day_bars
        universe.add(date, symbols)
        market_returns[date] = float(mkt_ret)

    return bars_by_date, universe, market_returns


def _sim_factor_matrix(
    close: np.ndarray,
    volume: np.ndarray,
    t: int,
    names: list[str],
) -> np.ndarray:
    """Compute simulator factors from price/volume history up to (excluding) day t.

    The simulator only drives returns with a small set of cheap-to-compute
    factors; the full 11-factor set is computed later (via polars) from the
    generated panel.  Returns an ``(n_stocks, len(names))`` matrix.

    Supported names: momentum_5d, momentum_20d, ma_dev_20, volatility_20d,
    volume_ratio.
    """
    n_stocks = close.shape[1]
    out = np.zeros((n_stocks, len(names)))
    for j, name in enumerate(names):
        if name == "momentum_5d":
            out[:, j] = (close[t - 1] - close[t - 6]) / close[t - 6]
        elif name == "momentum_20d":
            out[:, j] = (close[t - 1] - close[t - 21]) / close[t - 21]
        elif name == "ma_dev_20":
            sma = close[t - 20:t].mean(axis=0)
            out[:, j] = (close[t - 1] - sma) / sma
        elif name == "volatility_20d":
            rets = close[1:t] / close[:t - 1] - 1.0          # (t-1, N)
            out[:, j] = rets[t - 21:].std(axis=0, ddof=0)     # last 20 returns
        elif name == "volume_ratio":
            vsma = volume[t - 20:t].mean(axis=0)
            out[:, j] = volume[t - 1] / vsma
        else:
            raise ValueError(f"Unsupported simulator factor: {name}")
    return out


def bars_to_frame(bars_by_date: dict) -> "object":
    """Flatten ``{date: {symbol: DailyBar}}`` into a long-format polars DataFrame.

    One row per (symbol, date) with every ``DailyBar`` field as a column (the
    canonical ``schema.BAR_COLUMNS``).  This is the input shape for
    ``factors.compute_factor_panel``.
    """
    import polars as pl

    records = []
    for date, day in bars_by_date.items():
        for sym, b in day.items():
            # DailyBar field names match BAR_COLUMNS exactly, so drive the
            # record from the schema to keep the contract in one place.
            records.append({c: getattr(b, c) for c in BAR_COLUMNS})
    return pl.DataFrame(records)


def simulate_to_frame(n_stocks: int = 30, n_days: int = 300, seed: int = 42, **kwargs) -> "object":
    """Generate simulated data and return it as the canonical long table.

    This is the symmetric counterpart of ``ingest_parquet`` (real data): both
    return the same long-table schema, so the factor layer and everything
    downstream are source-agnostic.
    """
    bars, _, _ = generate_simulated_market(
        n_stocks=n_stocks, n_days=n_days, seed=seed, **kwargs)
    return bars_to_frame(bars)


def frame_to_bars(
    df: "object",
) -> tuple[dict[int, dict[str, "DailyBar"]], "StockUniverse", dict[int, float]]:
    """Convert a long-format polars DataFrame back into ``(bars, universe, mkt)``.

    This is the inverse of ``bars_to_frame`` — and the shape returned by
    ``generate_simulated_market`` — so ingested real data can be consumed by
    the same pipeline (engine, calibrator, portfolio) as simulated data.

    The DataFrame must have the canonical ``schema.BAR_COLUMNS``.
    """
    import polars as pl

    validate_bar_frame(df.columns)

    bars_by_date: dict[int, dict[str, DailyBar]] = {}
    universe = StockUniverse()
    market_returns: dict[int, float] = {}

    for (date,), grp in df.group_by("date", maintain_order=True):
        day: dict[str, DailyBar] = {}
        for row in grp.iter_rows(named=True):
            day[row["symbol"]] = DailyBar(
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
                market_return=float(row["market_return"]),
            )
        bars_by_date[int(date)] = day
        universe.add(int(date), list(day.keys()))
        market_returns[int(date)] = float(grp["market_return"][0])

    return bars_by_date, universe, market_returns
