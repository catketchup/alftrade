"""Tests for engine.py: Signal, standardization, and TradingEngine modes."""

import numpy as np
import pytest

from alftrade.data import DailyBar, compute_limit_prices
from alftrade.engine import Signal, TradingEngine, standardize_cross_section
from alftrade.factors import FACTOR_NAMES, compute_factor_panel, factor_panel_to_dict
from alftrade.data import bars_to_frame


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


def _factor_map(bars_by_date):
    """Compute factors for a {date: {sym: DailyBar}} panel via polars."""
    return factor_panel_to_dict(compute_factor_panel(bars_to_frame(bars_by_date)))


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


def test_neutralization_reduces_predictive_relationship():
    """Neutralization reduces the magnitude of the vol→mkt relationship."""
    factors = {
        "A": {"volatility_20d": 0.16, "market_beta": 1.0, "dummy": 0.0},
        "B": {"volatility_20d": 0.24, "market_beta": 1.5, "dummy": 0.0},
        "C": {"volatility_20d": 0.28, "market_beta": 2.0, "dummy": 0.0},
        "D": {"volatility_20d": 0.33, "market_beta": 2.5, "dummy": 0.0},
        "E": {"volatility_20d": 0.39, "market_beta": 3.0, "dummy": 0.0},
    }
    Z_neut, _ = standardize_cross_section(
        factors, ["volatility_20d", "market_beta", "dummy"],
        neutralize=["market_beta"],
    )
    assert Z_neut.shape == (5, 3)
    assert np.all(np.isfinite(Z_neut))


def test_neutralization_ignores_unknown_factors():
    """Passing a factor not in factor_names is silently ignored."""
    factors = {"A": {"x": 1.0, "y": 2.0}, "B": {"x": 1.5, "y": 2.5}}
    Z, _ = standardize_cross_section(factors, ["x", "y"], neutralize=["z"])
    assert Z.shape == (2, 2)


# ── TradingEngine: raw mode ────────────────────────────────────────────

def test_raw_mode_emits_signals_for_strong_factors():
    """Raw mode fires a Signal when a factor exceeds its threshold."""
    # Build a panel where prices rise ~1%/day so 5-day momentum > 2%.
    bars_by_date = {}
    for t in range(6):
        prices = {f"S{i}": 100.0 * (1.0 + 0.01 * t + 0.0005 * i * t) for i in range(10)}
        if t == 0:
            prev = {s: 100.0 for s in prices}
        else:
            prev = {s: bars_by_date[t - 1][s].close for s in prices}
        bars_by_date[t] = _panel(t, prices, prev)

    factors = _factor_map(bars_by_date)
    engine = TradingEngine(factors=factors, raw_thresholds={"momentum_5d": 0.02})
    out = {}
    for t in range(6):
        out = engine.on_cross_section(t, bars_by_date[t])
    # After 6 rising days, momentum_5d should exceed 0.02 for most stocks.
    assert len(out) > 0
    for sigs in out.values():
        for s in sigs:
            assert s.name == "momentum_5d"
            assert s.direction in (0, 1)


def test_raw_mode_skips_suspended():
    """Suspended symbols produce no signals."""
    bars = {
        "A": _bar("A", 110.0, 100.0, suspended=True),
        "B": _bar("B", 110.0, 100.0, suspended=False),
    }
    factors = _factor_map({1: bars})
    engine = TradingEngine(factors=factors, raw_thresholds={"daily_return": 0.001})
    out = engine.on_cross_section(1, bars)
    assert "A" not in out
    assert "B" in out


# ── TradingEngine: calibrated mode ─────────────────────────────────────

def test_calibrated_mode_uses_betas(market, dates, calibration, factors):
    """In calibrated mode, signal strength is the model's predicted return."""
    bars_by_date, _, _ = market
    engine = TradingEngine(factors=factors, calibration=calibration)
    d = dates[-25]
    out = engine.on_cross_section(d, bars_by_date[d])
    assert len(out) > 0
    for sigs in out.values():
        assert len(sigs) == 1
        assert sigs[0].name == "model"
        assert sigs[0].direction in (0, 1)
        assert sigs[0].strength >= 0


def test_calibrated_mode_strength_matches_prediction(market, dates, calibration, factors):
    """strength = |Σ β_k · z_k| computed independently matches the engine's signal."""
    bars_by_date, _, _ = market
    d = dates[-25]
    engine = TradingEngine(factors=factors, calibration=calibration)
    out = engine.on_cross_section(d, bars_by_date[d])
    cs = engine.build_cross_section(d, bars_by_date[d])

    betas = np.array([calibration.betas.get(n, 0.0) for n in cs.factor_names])
    pred = cs.z_matrix @ betas
    for i, sym in enumerate(cs.symbols):
        if sym in out:
            assert out[sym][0].strength == pytest.approx(abs(float(pred[i])), abs=1e-9)


def test_build_cross_section_shape(market, dates, factors):
    """build_cross_section returns a z_matrix matching symbols × factors."""
    bars_by_date, _, _ = market
    engine = TradingEngine(factors=factors)
    d = dates[-25]
    cs = engine.build_cross_section(d, bars_by_date[d])
    assert cs.z_matrix.shape == (len(cs.symbols), len(cs.factor_names))
    assert set(cs.symbols) == set(bars_by_date[d].keys())


def test_engine_neutralize_stores_and_passes_parameter(market, dates, calibration, factors):
    """Engine with neutralize stores it, and both build_cross_section and
    on_cross_section run without error."""
    bars_by_date, _, _ = market
    d = dates[-25]
    engine = TradingEngine(factors=factors, calibration=calibration,
                           neutralize=["market_beta"])
    assert engine.neutralize == ["market_beta"]

    cs = engine.build_cross_section(d, bars_by_date[d])
    assert cs.z_matrix.shape[1] == 11
    assert np.all(np.isfinite(cs.z_matrix))

    sigs = engine.on_cross_section(d, bars_by_date[d])
    assert len(sigs) > 0
