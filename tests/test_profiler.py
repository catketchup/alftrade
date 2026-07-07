"""Tests for profiler.py: FactorProfiler structure, ranges, and selection."""

import numpy as np
import pytest

from alftrade.factors import FACTOR_NAMES
from alftrade.calibration import FactorCalibrator
from alftrade.engine import TradingEngine
from alftrade.profiler import FactorProfile, ModelProfile, FactorProfiler


# ── Basic structure ────────────────────────────────────────────────────

def test_profile_returns_model_profile(market):
    """profile() returns a ModelProfile with per-factor profiles."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    prof = FactorProfiler.from_calibrator(cal)
    mp = prof.profile()
    assert isinstance(mp, ModelProfile)
    assert mp.n_samples == cal.num_samples
    assert mp.n_dates == cal.num_dates
    assert mp.n_factors == len(FACTOR_NAMES)
    assert len(mp.factor_profiles) == len(FACTOR_NAMES)
    for fp in mp.factor_profiles:
        assert isinstance(fp, FactorProfile)


def test_profile_has_all_factors(market):
    """Every FACTOR_NAMES entry appears exactly once in factor_profiles."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    names = [fp.name for fp in mp.factor_profiles]
    assert sorted(names) == sorted(FACTOR_NAMES)


# ── Ranges ─────────────────────────────────────────────────────────────

def test_ic_in_valid_range(market):
    """Every per-factor IC lies in [-1, 1]."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    for fp in mp.factor_profiles:
        assert -1.0 <= fp.ic <= 1.0
        assert -1.0 <= fp.ic_mean <= 1.0


def test_p_value_in_unit_interval(market):
    """p-values lie in [0, 1]."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    for fp in mp.factor_profiles:
        assert 0.0 <= fp.p_value <= 1.0


def test_coverage_in_unit_interval(market):
    """Coverage is a fraction in [0, 1]."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    for fp in mp.factor_profiles:
        assert 0.0 <= fp.coverage <= 1.0


def test_turnover_in_unit_interval(market):
    """Turnover (normalized rank change) lies in [0, 1]."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    for fp in mp.factor_profiles:
        assert 0.0 <= fp.turnover <= 1.0


def test_quantile_returns_length(market):
    """quantile_returns has exactly n_quantiles entries."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    prof = FactorProfiler.from_calibrator(cal, n_quantiles=5)
    mp = prof.profile()
    for fp in mp.factor_profiles:
        assert len(fp.quantile_returns) == 5


def test_monotonicity_in_valid_range(market):
    """Monotonicity (Spearman rho) lies in [-1, 1]."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    for fp in mp.factor_profiles:
        assert -1.0 <= fp.monotonicity <= 1.0
    assert -1.0 <= mp.model_monotonicity <= 1.0


# ── Predictability recovery ───────────────────────────────────────────

def test_strongest_factor_has_high_abs_ic(market):
    """volatility_20d (the simulator's strongest loaded factor) has |IC|>0.1."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    vol = next(fp for fp in mp.factor_profiles if fp.name == "volatility_20d")
    assert abs(vol.ic) > 0.1
    # And monotonic (low-vol → high return) → negative monotonicity.
    assert vol.monotonicity < 0


def test_model_monotonicity_positive(market):
    """Model predictions sort stocks in the right direction on average."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    assert mp.model_monotonicity > 0
    assert mp.model_ic > 0


# ── selected_factors ───────────────────────────────────────────────────

def test_selected_factors_filters_by_p_value(market):
    """selected_factors returns only factors below the p-value threshold."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    selected = mp.selected_factors(p_threshold=0.05)
    for fp in mp.factor_profiles:
        if fp.name in selected:
            assert fp.p_value < 0.05
        else:
            assert fp.p_value >= 0.05


def test_selected_factors_includes_strongest(market):
    """volatility_20d is significant and therefore selected at p<0.05."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    selected = mp.selected_factors(p_threshold=0.05)
    assert "volatility_20d" in selected


# ── repr ───────────────────────────────────────────────────────────────

def test_model_profile_repr_lists_factors(market):
    """repr lists every factor and the significance legend."""
    bars, _ = market
    dates = sorted(bars)
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cal.collect(d, engine.build_cross_section(d, bars[d]), bars)
    mp = FactorProfiler.from_calibrator(cal).profile()
    s = repr(mp)
    assert "ModelProfile" in s
    for name in FACTOR_NAMES:
        assert name in s
    assert "Significance" in s
