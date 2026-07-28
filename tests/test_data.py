"""Tests for data.py: DailyBar, price-limit helpers, and the simulator."""

import pytest
from alftrade.data import (
    DailyBar,
    StockUniverse,
    compute_limit_prices,
    round_price_to_tick,
    generate_simulated_market,
)


# ── Price-limit helpers ────────────────────────────────────────────────

def test_round_price_to_tick():
    """Prices round to the nearest 0.01 yuan."""
    assert round_price_to_tick(10.023) == pytest.approx(10.02)
    assert round_price_to_tick(10.026) == pytest.approx(10.03)
    assert round_price_to_tick(10.025) == pytest.approx(10.02)  # banker's rounding


def test_compute_limit_prices_main_board():
    """Main board ±10%: limit_up = prev_close*1.10, limit_down = prev_close*0.90."""
    up, down = compute_limit_prices(100.0, 0.10)
    assert up == pytest.approx(110.0)
    assert down == pytest.approx(90.0)


def test_compute_limit_prices_star_board():
    """STAR board ±20%."""
    up, down = compute_limit_prices(50.0, 0.20)
    assert up == pytest.approx(60.0)
    assert down == pytest.approx(40.0)


def test_compute_limit_prices_st_stocks():
    """ST stocks ±5%."""
    up, down = compute_limit_prices(10.0, 0.05)
    assert up == pytest.approx(10.5)
    assert down == pytest.approx(9.5)


def test_compute_limit_prices_zero_prev_close():
    """A zero prev_close returns (0, 0) rather than dividing by zero."""
    up, down = compute_limit_prices(0.0, 0.10)
    assert up == 0.0
    assert down == 0.0


# ── DailyBar ───────────────────────────────────────────────────────────

def _bar(close, prev_close=100.0, limit_pct=0.10, suspended=False):
    up, down = compute_limit_prices(prev_close, limit_pct)
    return DailyBar(
        date=1, symbol="000001.SZ",
        open=prev_close, high=close, low=close, close=close,
        volume=1000.0, amount=close * 1000.0,
        prev_close=prev_close, limit_up_price=up, limit_down_price=down,
        is_suspended=suspended,
    )


def test_bar_is_limit_up():
    """A close at the limit-up price is detected as limit-up."""
    bar = _bar(110.0, prev_close=100.0)
    assert bar.is_limit_up
    assert not bar.is_limit_down


def test_bar_is_limit_down():
    """A close at the limit-down price is detected as limit-down."""
    bar = _bar(90.0, prev_close=100.0)
    assert bar.is_limit_down
    assert not bar.is_limit_up


def test_bar_not_at_limit():
    """A close inside the band is neither limit-up nor limit-down."""
    bar = _bar(105.0, prev_close=100.0)
    assert not bar.is_limit_up
    assert not bar.is_limit_down


def test_bar_mid_price_is_close():
    """mid_price convenience property returns the close (execution convention)."""
    bar = _bar(100.0)
    assert bar.mid_price == 100.0


# ── StockUniverse ──────────────────────────────────────────────────────

def test_universe_active_on():
    """active_on returns the registered set; unknown dates return empty."""
    uni = StockUniverse()
    uni.add(20200101, ["A", "B"])
    uni.add(20200102, ["A", "C"])
    assert uni.active_on(20200101) == {"A", "B"}
    assert uni.active_on(20200102) == {"A", "C"}
    assert uni.active_on(19990101) == set()


def test_universe_symbols_and_dates():
    """symbols returns first-appearance order; dates returns ascending order."""
    uni = StockUniverse()
    uni.add(20200102, ["B", "C"])
    uni.add(20200101, ["A"])
    assert uni.symbols == ["B", "C", "A"]
    assert uni.dates == [20200101, 20200102]


# ── Simulator ──────────────────────────────────────────────────────────

def test_simulator_basic_shape():
    """Simulator produces a rectangular panel: every date has all symbols."""
    bars, uni, _ = generate_simulated_market(n_stocks=10, n_days=60, seed=1)
    dates = sorted(bars)
    syms = uni.symbols
    assert len(dates) == 60
    assert len(syms) == 10
    for d in dates:
        assert set(bars[d].keys()) == set(syms)


def test_simulator_prices_positive_and_within_limits():
    """All closes are positive and within the ±10% band of the prev close."""
    bars, _, _ = generate_simulated_market(n_stocks=10, n_days=80, seed=2)
    dates = sorted(bars)
    for d in dates[1:]:
        for bar in bars[d].values():
            assert bar.close > 0
            # Daily return must be within the limit band.
            ret = (bar.close - bar.prev_close) / bar.prev_close
            assert -0.1001 <= ret <= 0.1001


def test_simulator_reproducible():
    """Same seed → identical panel."""
    b1, _, _ = generate_simulated_market(n_stocks=5, n_days=40, seed=99)
    b2, _, _ = generate_simulated_market(n_stocks=5, n_days=40, seed=99)
    d = sorted(b1)[0]
    assert b1[d].keys() == b2[d].keys()
    for sym in b1[d]:
        assert b1[d][sym].close == pytest.approx(b2[d][sym].close)


def test_simulator_rejects_too_few_stocks():
    """A cross-sectional market needs at least 2 stocks."""
    with pytest.raises(ValueError):
        generate_simulated_market(n_stocks=1, n_days=50)


def test_simulator_rejects_too_few_days():
    """n_days must exceed burn_in + a margin."""
    with pytest.raises(ValueError):
        generate_simulated_market(n_stocks=5, n_days=35)  # burn_in=30 → 35 < 40
