"""Shared pytest fixtures for the alftrade test suite.

Generates a small in-memory A-share panel once per session, computes factors
with the polars vectorized engine, and reuses them across tests to keep the
suite fast. The panel has genuine factor structure (see
``alftrade.data.generate_simulated_market``) so calibration/profiler tests
have real signal to find.
"""

import pytest

from alftrade import (
    generate_simulated_market,
    bars_to_frame,
    compute_factor_panel,
    factor_panel_to_dict,
    TradingEngine,
    FactorCalibrator,
)


@pytest.fixture(scope="session")
def market():
    """A 20-stock × 150-day simulated A-share panel."""
    bars_by_date, universe, market_returns = generate_simulated_market(
        n_stocks=20, n_days=150, seed=11
    )
    return bars_by_date, universe, market_returns


@pytest.fixture(scope="session")
def dates(market):
    bars_by_date, _, _ = market
    return sorted(bars_by_date)


@pytest.fixture(scope="session")
def factors(market):
    """Precomputed factors {date: {symbol: {factor: value}}} via polars."""
    bars_by_date, _, _ = market
    long_df = bars_to_frame(bars_by_date)
    factor_df = compute_factor_panel(long_df)
    return factor_panel_to_dict(factor_df)


@pytest.fixture(scope="session")
def calibration(market, dates, factors):
    """A Fama-MacBeth CalibrationResult fit on the session panel.

    Uses a dedicated TradingEngine fed with the precomputed polars factors.
    """
    engine = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cs = engine.build_cross_section(d, market[0][d])
        cal.collect(d, cs, market[0])
    return cal.fit()
