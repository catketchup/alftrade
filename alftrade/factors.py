"""
factors.py - Cross-sectional daily factor calculator.

This is the daily-frequency, pure-Python analogue of hftrade's C++
``FactorCalculator``. Two key differences from the high-frequency version:

  1. **No C++/Cython.** Daily frequency means we process at most one bar per
     stock per day; numpy rolling windows are more than fast enough and the
     build stays a pure-Python ``pip install``.

  2. **All factors are already stationary.** hftrade had to transform raw
     price levels (mid_price ~100, vwap ~100) into stationary ratios before
     regressing them against returns. Here every factor is *defined* as a
     return, ratio, or bounded indicator, so there is no stationarity
     transform inside the calculator itself. Cross-sectional standardization
     (z-scoring within each date's universe) happens later, in
     ``calibration.transform_factors``, because it requires the whole
     cross-section — not a single stock.

Factor list (10 factors, all computable from OHLCV):
    daily_return      = (close - prev_close) / prev_close
    overnight_return  = (open  - prev_close) / prev_close
    intraday_return   = (close - open)       / open
    momentum_5d       = (close - close[t-5])  / close[t-5]
    momentum_20d      = (close - close[t-20]) / close[t-20]
    ma_dev_20         = (close - SMA20)       / SMA20
    volatility_20d    = std(daily_return, 20)
    volume_ratio      = volume / SMA20(volume)
    rsi_14            = RSI(14) on close, in [0, 100] (rescaled to [-1, 1])
    atr_pct_14        = ATR(14) / close

Usage:
    calc = FactorCalculator()
    for bar in bars_for_one_symbol:
        factors = calc.update(bar)   # advances rolling state, returns latest
    # or: calc.latest_factors  -> dict of the most recent factor values

The calculator keeps per-stock rolling state (price/volume windows, running
returns). Create **one instance per symbol**, exactly like hftrade's
``FactorCalculator(sma_window)`` was one per instrument.
"""

from collections import deque
from typing import Optional

from alftrade.data import DailyBar


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
]


def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    """Division that returns `fallback` when the divisor is ~0."""
    return num / den if abs(den) > 1e-12 else fallback


class FactorCalculator:
    """Per-symbol rolling daily factor calculator.

    Maintains:
      * a ring buffer of recent closes (for momentum / SMA / RSI),
      * a ring buffer of recent daily returns (for volatility),
      * a ring buffer of recent volumes (for volume_ratio),
      * running True Range accumulators (for ATR).

    All factors are O(1) amortized per bar thanks to the deque sliding
    windows — the same design hftrade's C++ calculator used for SMA/VWAP.

    Args:
        sma_window: SMA lookback in trading days (default 20).
        vol_window: Volatility lookback in trading days (default 20).
        rsi_window: RSI lookback (default 14).
        atr_window: ATR lookback (default 14).
    """

    def __init__(
        self,
        sma_window: int = 20,
        vol_window: int = 20,
        rsi_window: int = 14,
        atr_window: int = 14,
    ):
        self.sma_window = sma_window
        self.vol_window = vol_window
        self.rsi_window = rsi_window
        self.atr_window = atr_window

        self._closes: deque[float] = deque(maxlen=max(sma_window, 20) + 1)
        self._returns: deque[float] = deque(maxlen=vol_window)
        self._volumes: deque[float] = deque(maxlen=sma_window)
        # RSI uses Wilder smoothing: track avg gain / avg loss.
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._rsi_count: int = 0
        self._prev_close: Optional[float] = None
        # ATR via Wilder smoothing of True Range.
        self._tr_sum: float = 0.0
        self._atr: Optional[float] = None
        self._atr_count: int = 0

        self._latest: dict[str, float] = {name: 0.0 for name in FACTOR_NAMES}

    @property
    def latest_factors(self) -> dict[str, float]:
        """Most recent factor values as a dict (0.0 until enough history)."""
        return dict(self._latest)

    def update(self, bar: DailyBar) -> dict[str, float]:
        """Ingest one daily bar and return the updated factor dict.

        This both advances the rolling state and returns the latest factors,
        so callers can write ``factors = calc.update(bar)`` in a single
        expression. Mirrors hftrade's ``PyFactorCalculator.compute(...)`` but
        takes a ``DailyBar`` instead of scalar tick fields.
        """
        close = bar.close
        prev_close = bar.prev_close if bar.prev_close > 0 else (
            self._closes[-1] if len(self._closes) >= 1 else close
        )

        # ── Price-based factors ──
        daily_return = _safe_div(close - prev_close, prev_close)
        overnight_return = _safe_div(bar.open - prev_close, prev_close)
        intraday_return = _safe_div(close - bar.open, bar.open)

        self._closes.append(close)
        self._volumes.append(bar.volume)

        sma = (sum(self._closes) / len(self._closes)) if self._closes else close
        ma_dev_20 = _safe_div(close - sma, sma)

        # Momentum: need closes from 5 / 20 bars ago.
        momentum_5d = self._momentum(5)
        momentum_20d = self._momentum(20)

        # ── Volatility: rolling std of daily returns ──
        self._returns.append(daily_return)
        volatility_20d = self._rolling_std(self._returns)

        # ── Volume ratio ──
        vol_sma = (sum(self._volumes) / len(self._volumes)) if self._volumes else bar.volume
        volume_ratio = _safe_div(bar.volume, vol_sma, 1.0)

        # ── RSI(14) via Wilder smoothing, rescaled to [-1, 1] ──
        rsi_raw = self._update_rsi(daily_return)
        rsi_14 = (rsi_raw - 50.0) / 50.0  # 0..100 -> -1..+1

        # ── ATR(14) as a fraction of close ──
        atr = self._update_atr(bar)
        atr_pct_14 = _safe_div(atr, close)

        self._latest = {
            "daily_return": daily_return,
            "overnight_return": overnight_return,
            "intraday_return": intraday_return,
            "momentum_5d": momentum_5d,
            "momentum_20d": momentum_20d,
            "ma_dev_20": ma_dev_20,
            "volatility_20d": volatility_20d,
            "volume_ratio": volume_ratio,
            "rsi_14": rsi_14,
            "atr_pct_14": atr_pct_14,
        }
        return dict(self._latest)

    # ── Rolling-helpers ─────────────────────────────────────────────

    def _momentum(self, n: int) -> float:
        """Return (close - close[t-n]) / close[t-n], or 0 if not enough history."""
        if len(self._closes) <= n:
            return 0.0
        old = self._closes[-(n + 1)]
        return _safe_div(self._closes[-1] - old, old)

    @staticmethod
    def _rolling_std(values: deque) -> float:
        """Population std of the deque's contents (0 if < 2 points)."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        return var ** 0.5

    def _update_rsi(self, change: float) -> float:
        """Advance Wilder RSI state and return RSI in [0, 100].

        Wilder smoothing:
          First rsi_window changes: simple averages of gains/losses.
          Thereafter: avg = (prev_avg * (n-1) + current) / n.
        Returns 50.0 (neutral) until enough history has accumulated.

        During the bootstrap window (calls 1..w), ``_avg_gain`` / ``_avg_loss``
        are used as running *sums*; at call w they are divided by w to become
        averages, after which Wilder smoothing takes over.
        """
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        w = self.rsi_window

        if self._avg_gain is None:
            # First change: start the running sums.
            self._avg_gain = gain
            self._avg_loss = loss
            self._rsi_count = 1
            return 50.0
        self._rsi_count += 1
        if self._rsi_count <= w:
            # Still bootstrapping: accumulate running sums.
            self._avg_gain += gain
            self._avg_loss += loss
            if self._rsi_count == w:
                self._avg_gain /= w
                self._avg_loss /= w
            return 50.0
        # Wilder smoothing after bootstrap.
        self._avg_gain = (self._avg_gain * (w - 1) + gain) / w
        self._avg_loss = (self._avg_loss * (w - 1) + loss) / w
        if self._avg_loss < 1e-12:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _update_atr(self, bar: DailyBar) -> float:
        """Advance Wilder ATR state and return the current ATR value.

        True Range = max(
            high - low,
            |high - prev_close|,
            |low  - prev_close|
        )
        Bootstrap: average of first `atr_window` TRs.
        Thereafter: ATR = (prev_ATR * (n-1) + TR) / n.
        Returns 0.0 until the first bar is seen.
        """
        prev_close = bar.prev_close if bar.prev_close > 0 else (
            self._closes[-2] if len(self._closes) >= 2 else bar.close
        )
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_close),
            abs(bar.low - prev_close),
        )
        w = self.atr_window
        self._atr_count += 1
        if self._atr is None:
            self._tr_sum = tr
            self._atr = tr
            return tr
        if self._atr_count <= w:
            self._tr_sum += tr
            if self._atr_count == w:
                self._atr = self._tr_sum / w
            else:
                self._atr = self._tr_sum / self._atr_count  # provisional avg
            return self._atr
        # Wilder smoothing.
        self._atr = (self._atr * (w - 1) + tr) / w
        return self._atr

    def reset(self) -> None:
        """Clear all rolling state (e.g. before reusing on a new symbol)."""
        self._closes.clear()
        self._returns.clear()
        self._volumes.clear()
        self._avg_gain = None
        self._avg_loss = None
        self._rsi_count = 0
        self._prev_close = None
        self._tr_sum = 0.0
        self._atr = None
        self._atr_count = 0
        self._latest = {name: 0.0 for name in FACTOR_NAMES}


def compute_cross_section(
    bars: dict[str, DailyBar],
    calcs: dict[str, FactorCalculator],
) -> dict[str, dict[str, float]]:
    """Feed one day's bars into per-symbol calculators; return {sym: factors}.

    Convenience wrapper used by the engine. ``calcs`` must contain a
    ``FactorCalculator`` for every symbol in ``bars``; each calculator is
    advanced with that symbol's bar.

    Args:
        bars: {symbol: DailyBar} for a single trading date.
        calcs: {symbol: FactorCalculator} with one calculator per symbol,
               reused across dates (so rolling state persists day to day).

    Returns:
        {symbol: {factor_name: value}} — the latest factors per symbol.
    """
    out: dict[str, dict[str, float]] = {}
    for sym, bar in bars.items():
        calc = calcs.get(sym)
        if calc is None:
            calc = FactorCalculator()
            calcs[sym] = calc
        out[sym] = calc.update(bar)
    return out
