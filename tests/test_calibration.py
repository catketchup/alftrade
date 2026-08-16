"""Tests for calibration.py: forward_return, collect/fit, persistence,
and recovery of known factor structure.
"""

import os
import json
import tempfile

import numpy as np
import pytest

from alftrade.data import DailyBar
from alftrade.calibration import (
    CalibrationResult, FactorCalibrator, forward_return, _CloseOnly,
)
from alftrade.engine import TradingEngine
from alftrade.factors import FACTOR_NAMES


# ── forward_return ─────────────────────────────────────────────────────

def test_forward_return_basic():
    """y = (close[t+N] - close[t]) / close[t]."""
    bar_t = _CloseOnly(close=100.0)
    bar_tN = DailyBar(date=2, symbol="A", open=0, high=0, low=0, close=110.0,
                      volume=0, amount=0, prev_close=100.0)
    assert forward_return(bar_t, bar_tN) == pytest.approx(0.10)


def test_forward_return_zero_close():
    """A zero close on day t returns 0 (guarded against division by zero)."""
    bar_t = _CloseOnly(close=0.0)
    bar_tN = DailyBar(date=2, symbol="A", open=0, high=0, low=0, close=110.0,
                      volume=0, amount=0, prev_close=100.0)
    assert forward_return(bar_t, bar_tN) == 0.0


# ── collect / fit ──────────────────────────────────────────────────────

def test_calibrator_not_ready_with_little_data(market, factors):
    """With too few samples, ready is False and fit raises."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=10_000, min_dates=10_000)
    for d in dates[:5]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    assert not cal.ready
    with pytest.raises(ValueError):
        cal.fit()


def test_fit_returns_calibration_result(market, factors):
    """fit() on enough data returns a populated CalibrationResult."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit()
    assert isinstance(result, CalibrationResult)
    assert result.horizon == 5
    assert result.sample_size > 0
    assert result.n_dates > 0
    assert set(result.factor_names) == set(FACTOR_NAMES)
    assert set(result.betas.keys()) == set(FACTOR_NAMES)
    assert set(result.ics.keys()) == set(FACTOR_NAMES)


def test_fit_fama_macbeth_has_t_stats(market, factors):
    """FM fit produces finite t-stats and p-values in [0, 1]."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit(method="fama_macbeth")
    assert result.method == "fama_macbeth"
    for name in FACTOR_NAMES:
        assert np.isfinite(result.fm_t_stats[name])
        assert 0.0 <= result.fm_p_values[name] <= 1.0


def test_fit_pooled_has_no_t_stats(market, factors):
    """Pooled fit leaves FM t-stats as NaN."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit(method="pooled")
    assert result.method == "pooled"
    for name in FACTOR_NAMES:
        assert not np.isfinite(result.fm_t_stats[name])


def test_fit_recovers_loaded_factor_sign(market, factors):
    """The simulator's strongest loaded factor (volatility, β<0) is
    recovered with the correct sign and high significance."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit()
    # volatility_20d has a strong negative loading in DEFAULT_FACTOR_LOADINGS.
    assert result.betas["volatility_20d"] < 0
    assert result.fm_t_stats["volatility_20d"] < -2.0   # significant
    assert result.ics["volatility_20d"] < 0


def test_larger_alpha_shrinks_betas(market, factors):
    """A larger Ridge alpha shrinks β toward zero (regularization)."""
    bars, _, _ = market
    dates = sorted(bars)

    def fit_with(alpha):
        eng = TradingEngine(factors=factors)
        cal = FactorCalibrator(horizon=5, alpha=alpha, min_samples=50, min_dates=10)
        for d in dates[:-20]:
            cal.collect(d, eng.build_cross_section(d, bars[d]), bars)
        return cal.fit()

    small = fit_with(alpha=0.01)
    large = fit_with(alpha=1000.0)
    # The sum of |β| should be smaller with heavy regularization.
    assert sum(abs(v) for v in large.betas.values()) < sum(abs(v) for v in small.betas.values())


def test_num_samples_and_dates_grow(market, factors):
    """collect() increases num_samples and num_dates over time."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    before = cal.num_samples
    for d in dates[:30]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    assert cal.num_samples > before
    assert cal.num_dates > 0


def test_reset_clears_data():
    """reset() empties both the buffer and the by_date collection."""
    cal = FactorCalibrator(horizon=5)
    cal._buffer.append((1, ["A"], np.zeros((1, 1)), {"A": 100.0}))
    cal._by_date[1] = (np.zeros((1, 1)), np.zeros(1))
    cal.reset()
    assert len(cal._buffer) == 0
    assert len(cal._by_date) == 0


# ── Persistence ────────────────────────────────────────────────────────

def test_save_load_roundtrip(market, tmp_path, factors):
    """save() then load() restores every field exactly."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit()
    path = str(tmp_path / "calibration.json")
    result.save(path)
    loaded = CalibrationResult.load(path)
    assert loaded.horizon == result.horizon
    assert loaded.method == result.method
    assert loaded.factor_names == result.factor_names
    assert loaded.betas == result.betas
    assert loaded.ics == result.ics
    assert loaded.sample_size == result.sample_size
    assert loaded.n_dates == result.n_dates


def test_save_produces_valid_json(market, tmp_path, factors):
    """The saved file is parseable JSON with the expected top-level keys."""
    bars, _, _ = market
    dates = sorted(bars)
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    result = cal.fit()
    path = str(tmp_path / "calibration.json")
    result.save(path)
    with open(path) as f:
        data = json.load(f)
    for key in ("horizon", "alpha", "method", "factor_names", "betas",
                "ics", "r_squared", "sample_size", "n_dates"):
        assert key in data


# ── CalibrationResult repr ─────────────────────────────────────────────

def test_calibration_result_repr(calibration):
    """repr is informative and lists every factor."""
    s = repr(calibration)
    assert "CalibrationResult" in s
    for name in FACTOR_NAMES:
        assert name in s
