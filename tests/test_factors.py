"""Tests for factors.py: FactorCalculator and compute_cross_section.

Each test feeds a synthetic OHLCV stream into a FactorCalculator and checks
the computed factor values against hand-computed expectations.
"""

import pytest
from alftrade.data import DailyBar, compute_limit_prices
from alftrade.factors import FactorCalculator, FACTOR_NAMES, compute_cross_section


def _bar(date, close, prev_close, open_=None, high=None, low=None,
         volume=1000.0, total_shares=1e8):
    """Build a minimal DailyBar with auto OHLC and price-limit metadata."""
    o = open_ if open_ is not None else prev_close
    h = high if high is not None else max(o, close) * 1.001
    l = low if low is not None else min(o, close) * 0.999
    up, dn = compute_limit_prices(prev_close, 0.10)
    return DailyBar(
        date=date, symbol="TEST",
        open=o, high=h, low=l, close=close,
        volume=volume, amount=close * volume,
        prev_close=prev_close, limit_up_price=up, limit_down_price=dn,
        total_shares=total_shares,
    )


# ── FactorCalculator basics ────────────────────────────────────────────

def test_factor_names_canonical():
    """FACTOR_NAMES has the 11 expected factors in order."""
    assert FACTOR_NAMES == [
        "daily_return", "overnight_return", "intraday_return",
        "momentum_5d", "momentum_20d", "ma_dev_20", "volatility_20d",
        "volume_ratio", "rsi_14", "atr_pct_14", "market_beta",
    ]


def test_latest_factors_before_update():
    """A fresh calculator returns all factors as 0.0."""
    calc = FactorCalculator()
    f = calc.latest_factors
    assert set(f.keys()) == set(FACTOR_NAMES)
    assert all(v == 0.0 for v in f.values())


def test_daily_return():
    """daily_return = (close - prev_close) / prev_close."""
    calc = FactorCalculator()
    f = calc.update(_bar(1, close=105.0, prev_close=100.0))
    assert f["daily_return"] == pytest.approx(0.05)


def test_overnight_return():
    """overnight_return = (open - prev_close) / prev_close."""
    calc = FactorCalculator()
    f = calc.update(_bar(1, close=105.0, prev_close=100.0, open_=102.0))
    assert f["overnight_return"] == pytest.approx(0.02)


def test_intraday_return():
    """intraday_return = (close - open) / open."""
    calc = FactorCalculator()
    f = calc.update(_bar(1, close=105.0, prev_close=100.0, open_=102.0))
    assert f["intraday_return"] == pytest.approx((105.0 - 102.0) / 102.0)


def test_momentum_5d():
    """momentum_5d = (close - close[t-5]) / close[t-5] after 6 bars."""
    calc = FactorCalculator()
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
    prev = 100.0
    for i, p in enumerate(prices):
        f = calc.update(_bar(i + 1, close=p, prev_close=prev))
        prev = p
    # After 6 bars, momentum_5d = (110 - 100) / 100 = 0.10
    assert f["momentum_5d"] == pytest.approx(0.10)


def test_momentum_5d_zero_before_enough_history():
    """momentum_5d is 0 until 6 bars have been seen."""
    calc = FactorCalculator()
    f = calc.update(_bar(1, close=100.0, prev_close=100.0))
    assert f["momentum_5d"] == 0.0
    f = calc.update(_bar(2, close=101.0, prev_close=100.0))
    assert f["momentum_5d"] == 0.0


def test_ma_dev_20():
    """ma_dev_20 = (close - SMA20) / SMA20."""
    calc = FactorCalculator(sma_window=20)
    # Feed 20 bars all at 100, then one at 110.
    prev = 100.0
    for i in range(20):
        f = calc.update(_bar(i + 1, close=100.0, prev_close=prev))
        prev = 100.0
    f = calc.update(_bar(21, close=110.0, prev_close=100.0))
    # SMA20 over the window of 21 bars: deque maxlen is 21, so all 21 closes
    # are in the window → SMA = (20*100 + 110)/21
    sma = (20 * 100.0 + 110.0) / 21
    assert f["ma_dev_20"] == pytest.approx((110.0 - sma) / sma)


def test_volatility_20d_nonneg():
    """volatility_20d is a std, so it is ≥ 0, and > 0 with varying returns."""
    calc = FactorCalculator()
    # Alternating up/down returns produce nonzero volatility.
    prev = 100.0
    for i in range(20):
        p = prev * (1.01 if i % 2 == 0 else -0.005 + 1)
        f = calc.update(_bar(i + 1, close=p, prev_close=prev))
        prev = p
    assert f["volatility_20d"] > 0


def test_volume_ratio():
    """volume_ratio = today_volume / SMA(volume, window)."""
    calc = FactorCalculator()
    # First 3 bars at volume 1000 → SMA = 1000; 4th bar at 2000 → ratio = 2.0
    # but the deque maxlen is 20, so after 4 bars SMA = (1000*3 + 2000)/4 = 1250
    # and ratio = 2000/1250 = 1.6.
    prev = 100.0
    for i in range(3):
        f = calc.update(_bar(i + 1, close=100.0, prev_close=prev, volume=1000.0))
        prev = 100.0
    f = calc.update(_bar(4, close=100.0, prev_close=prev, volume=2000.0))
    assert f["volume_ratio"] == pytest.approx(2000.0 / 1250.0)


def test_rsi_14_range_and_neutrality():
    """RSI is in [-1, 1] after rescaling and starts near 0 (neutral=50)."""
    calc = FactorCalculator(rsi_window=14)
    prev = 100.0
    for i in range(20):
        p = prev * (1.0 + (0.001 if i % 2 else -0.001))
        f = calc.update(_bar(i + 1, close=p, prev_close=prev))
        prev = p
    assert -1.0 <= f["rsi_14"] <= 1.0


def test_rsi_14_all_up_moves_positive():
    """A pure up-trend pushes RSI toward +1 (RSI raw → 100)."""
    calc = FactorCalculator(rsi_window=14)
    prev = 100.0
    for i in range(30):
        p = prev * 1.01
        f = calc.update(_bar(i + 1, close=p, prev_close=prev))
        prev = p
    assert f["rsi_14"] > 0.9  # near +1


def test_atr_pct_14_positive():
    """ATR as a fraction of close is positive once high-low spreads exist."""
    calc = FactorCalculator(atr_window=14)
    prev = 100.0
    for i in range(20):
        f = calc.update(_bar(i + 1, close=100.0, prev_close=prev,
                             high=101.0, low=99.0))
        prev = 100.0
    assert f["atr_pct_14"] > 0


def test_reset_clears_state():
    """After reset, factors revert to 0 and rolling windows start over."""
    calc = FactorCalculator()
    for i in range(10):
        calc.update(_bar(i + 1, close=100.0 + i, prev_close=100.0 + i - 1))
    assert calc.latest_factors["momentum_5d"] != 0.0
    calc.reset()
    assert all(v == 0.0 for v in calc.latest_factors.values())
    # First bar after reset: momentum_5d should be 0 (no history).
    f = calc.update(_bar(1, close=100.0, prev_close=100.0))
    assert f["momentum_5d"] == 0.0


# ── compute_cross_section ──────────────────────────────────────────────

def test_compute_cross_section_returns_per_symbol():
    """compute_cross_section advances every calculator and returns factors."""
    bar1 = _bar(1, close=100.0, prev_close=99.0)
    bar1.symbol = "A"
    bar2 = _bar(1, close=200.0, prev_close=198.0)
    bar2.symbol = "B"
    calcs = {}
    out = compute_cross_section({"A": bar1, "B": bar2}, calcs)
    assert set(out.keys()) == {"A", "B"}
    assert out["A"]["daily_return"] == pytest.approx((100.0 - 99.0) / 99.0)
    assert out["B"]["daily_return"] == pytest.approx((200.0 - 198.0) / 198.0)
    # Calculators are stored for reuse across dates.
    assert set(calcs.keys()) == {"A", "B"}


# ── market_beta factor ──────────────────────────────────────────────

def test_market_beta_defaults_to_neutral():
    """With no market returns, market_beta stays at 1.0 (neutral)."""
    calc = FactorCalculator()
    # Feed several bars with market_return=0.
    prev = 100.0
    for i in range(30):
        # _bar helper doesn't set market_return; it defaults to 0.
        f = calc.update(_bar(i + 1, close=100.0 + i * 0.1, prev_close=prev))
        prev = 100.0 + i * 0.1
    # When market_return is always 0, cov=0 → beta=1.0 (fallback).
    assert f["market_beta"] == 1.0


def test_market_beta_recovers_known_loading():
    """When daily_return = 0.5 × market_return + noise, beta ≈ 0.5."""
    calc = FactorCalculator(beta_window=30)
    prev = 100.0
    for i in range(40):
        mkt = (i % 3 - 1) * 0.01  # alternating market
        # stock return = 1.2 × mkt + noise → true beta = 1.2
        noise = (i % 7 - 3) * 0.002
        ret = 1.2 * mkt + noise
        close = prev * (1.0 + ret)
        bar = _bar(i + 1, close=close, prev_close=prev)
        bar.market_return = mkt
        f = calc.update(bar)
        prev = close
    # After 40 bars with 30-day window, beta should be ~1.2.
    assert 0.8 <= f["market_beta"] <= 1.6


def test_compute_cross_section_creates_missing_calculator():
    """A symbol without a pre-existing calculator gets a new one."""
    bar = _bar(1, close=100.0, prev_close=100.0)
    bar.symbol = "NEW"
    calcs = {}
    out = compute_cross_section({"NEW": bar}, calcs)
    assert "NEW" in calcs
    assert out["NEW"]["daily_return"] == pytest.approx(0.0)
