"""Shared pytest fixtures for the alftrade test suite.

Generates a small in-memory A-share panel once per session and reuses it
across tests to keep the suite fast. The panel has genuine factor structure
(see ``alftrade.data.generate_simulated_market``) so calibration/profiler
tests have real signal to find.
"""

import pytest

from alftrade import generate_simulated_market, TradingEngine, FactorCalibrator


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
def calibration(market, dates):
    """A Fama-MacBeth CalibrationResult fit on the session panel.

    Built with a dedicated offline TradingEngine so its per-symbol factor
    state is independent of any test's engine.
    """
    engine = TradingEngine()
    cal = FactorCalibrator(horizon=5, min_samples=50, min_dates=10)
    for d in dates[:-20]:
        cs = engine.build_cross_section(d, market[0][d])
        cal.collect(d, cs, market[0])
    return cal.fit()
