"""Tests for engine.py: Signal, standardization, and TradingEngine modes."""

import numpy as np
import pytest

from alftrade.data import DailyBar, compute_limit_prices
from alftrade.engine import Signal, TradingEngine, standardize_cross_section
from alftrade.factors import FACTOR_NAMES


def _bar(sym, close, prev_close, suspended=False):
    up, dn = compute_limit_prices(prev_close, 0.10)
    return DailyBar(
        date=1, symbol=sym, open=prev_close, high=close, low=close, close=close,
        volume=1000.0, amount=close * 1000.0, prev_close=prev_close,
        limit_up_price=up, limit_down_price=dn, is_suspended=suspended,
    )


def _panel(date, prices, prev_closes=None, suspended=None):
    """Build {sym: DailyBar} from a dict of close prices."""
    syms = list(prices.keys())
    if prev_closes is None:
        prev_closes = {s: prices[s] for s in syms}
    if suspended is None:
        suspended = {s: False for s in syms}
    return {
        s: _bar(s, prices[s], prev_closes.get(s, prices[s]), suspended[s])
        for s in syms
    }


# ── Signal ─────────────────────────────────────────────────────────────

def test_signal_repr():
    """Signal repr encodes direction as LONG/FLAT."""
    s = Signal("000001.SZ", "model", 1, 0.05)
    assert "LONG" in repr(s)
    assert "000001.SZ" in repr(s)
    s2 = Signal("000001.SZ", "model", 0, 0.0)
    assert "FLAT" in repr(s2)


# ── standardize_cross_section ──────────────────────────────────────────

def test_standardize_mean_zero_unit_std():
    """Z-scoring produces columns with ~0 mean and ~1 std."""
    factors = {
        "A": {"daily_return": 0.01, "overnight_return": 0.0},
        "B": {"daily_return": 0.02, "overnight_return": 0.0},
        "C": {"daily_return": 0.03, "overnight_return": 0.0},
    }
    Z, syms = standardize_cross_section(factors, ["daily_return", "overnight_return"])
    assert syms == ["A", "B", "C"]
    assert Z.shape == (3, 2)
    assert np.mean(Z[:, 0]) == pytest.approx(0.0, abs=1e-9)
    assert np.std(Z[:, 0]) == pytest.approx(1.0, abs=1e-9)


def test_standardize_constant_factor_to_zero():
    """A factor with no cross-sectional dispersion maps to all zeros."""
    factors = {
        "A": {"x": 5.0, "y": 0.01},
        "B": {"x": 5.0, "y": 0.02},
        "C": {"x": 5.0, "y": 0.03},
    }
    Z, _ = standardize_cross_section(factors, ["x", "y"])
    assert np.all(Z[:, 0] == 0.0)
    assert np.std(Z[:, 1]) == pytest.approx(1.0, abs=1e-9)


# ── TradingEngine: raw mode ────────────────────────────────────────────

def test_raw_mode_emits_signals_for_strong_factors():
    """Raw mode fires a Signal when a factor exceeds its threshold."""
    engine = TradingEngine(raw_thresholds={"momentum_5d": 0.02})
    # Build a panel where prices rise ~1%/day so 5-day momentum > 2%.
    bars_series = []
    for t in range(6):
        prices = {f"S{i}": 100.0 * (1.0 + 0.01 * t + 0.0005 * i * t) for i in range(10)}
        if t == 0:
            prev = {s: 100.0 for s in prices}
        else:
            prev = {s: bars_series[-1][s].close for s in prices}
        bars_series.append(_panel(t, prices, prev))
    # Feed days 0..5 through the engine (advances its internal calculators).
    out = {}
    for t, bars in enumerate(bars_series):
        out = engine.on_cross_section(t, bars)
    # After 6 rising days, momentum_5d should exceed 0.02 for most stocks.
    assert len(out) > 0
    for sigs in out.values():
        for s in sigs:
            assert s.name == "momentum_5d"
            assert s.direction in (0, 1)


def test_raw_mode_skips_suspended():
    """Suspended symbols produce no signals."""
    engine = TradingEngine(raw_thresholds={"daily_return": 0.001})
    bars = {
        "A": _bar("A", 110.0, 100.0, suspended=True),   # would fire but suspended
        "B": _bar("B", 110.0, 100.0, suspended=False),
    }
    out = engine.on_cross_section(1, bars)
    assert "A" not in out
    assert "B" in out


def test_engine_reset_clears_state():
    """reset() drops per-symbol factor calculators."""
    engine = TradingEngine()
    engine.on_cross_section(1, {"A": _bar("A", 100.0, 100.0)})
    assert "A" in engine._calcs
    engine.reset()
    assert engine._calcs == {}


# ── TradingEngine: calibrated mode ─────────────────────────────────────

def test_calibrated_mode_uses_betas(market, dates, calibration):
    """In calibrated mode, signal strength is the model's predicted return."""
    bars_by_date, _ = market
    engine = TradingEngine(calibration=calibration)
    # Pick a date late enough that factors are well-defined.
    d = dates[-25]
    out = engine.on_cross_section(d, bars_by_date[d])
    assert len(out) > 0
    # Every emitted signal should be the single "model" signal.
    for sigs in out.values():
        assert len(sigs) == 1
        assert sigs[0].name == "model"
        assert sigs[0].direction in (0, 1)
        assert sigs[0].strength >= 0


def test_calibrated_mode_strength_matches_prediction(market, dates, calibration):
    """strength = |Σ β_k · z_k| computed independently matches the engine's signal.

    Two fresh engines replay the same history: one via on_cross_section
    (produces signals), one via build_cross_section (produces the z_matrix).
    Both advance their per-symbol calculators identically, so their factor
    states match and the hand-computed prediction must equal the signal
    strength.
    """
    bars_by_date, _ = market
    d = dates[-25]

    # Engine 1: replay history, capturing signals on date d.
    engine = TradingEngine(calibration=calibration)
    out = None
    for dd in dates:
        if dd > d:
            break
        out = engine.on_cross_section(dd, bars_by_date[dd])

    # Engine 2: replay history, capturing the cross-section on date d.
    eng2 = TradingEngine(calibration=calibration)
    cs = None
    for dd in dates:
        if dd > d:
            break
        cs = eng2.build_cross_section(dd, bars_by_date[dd])

    assert cs is not None and out is not None
    betas = np.array([calibration.betas.get(n, 0.0) for n in cs.factor_names])
    pred = cs.z_matrix @ betas
    for i, sym in enumerate(cs.symbols):
        if sym in out:
            expected = abs(float(pred[i]))
            assert out[sym][0].strength == pytest.approx(expected, abs=1e-9)


def test_build_cross_section_shape(market, dates):
    """build_cross_section returns a z_matrix matching symbols × factors."""
    bars_by_date, _ = market
    engine = TradingEngine()
    d = dates[-25]
    cs = engine.build_cross_section(d, bars_by_date[d])
    assert cs.z_matrix.shape == (len(cs.symbols), len(cs.factor_names))
    assert set(cs.symbols) == set(bars_by_date[d].keys())
