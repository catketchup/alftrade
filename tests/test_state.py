"""Tests for state serialization: Position / Portfolio / RiskManager /
PortfolioManager / CalibrationResult / Trader round-trips."""

import json

import pytest

from alftrade import (
    TradingEngine, PortfolioManager, SignalAggregator, Broker, ExecutionConfig,
    Trader, FactorCalibrator,
)
from alftrade.calibration import CalibrationResult
from alftrade.portfolio import Position, Portfolio, Trade, CostModel
from alftrade.risk import RiskConfig, RiskManager


def test_trade_round_trip():
    t = Trade(20240101, "600000.SH", 10.5, 100, "OPEN", cost=5.0, pnl=-5.0, reason="REBALANCE")
    t2 = Trade.from_dict(t.to_dict())
    assert t2 == t


def test_position_round_trip():
    pos = Position(quantity=500, avg_entry_price=10.0, t1_locked=300, realized_pnl=12.5)
    pos2 = Position.from_state(pos.to_state())
    assert pos2.quantity == 500
    assert pos2.avg_entry_price == pytest.approx(10.0)
    assert pos2.t1_locked == 300
    assert pos2.realized_pnl == pytest.approx(12.5)
    assert pos2.available_qty == pytest.approx(200)


def test_portfolio_round_trip():
    pf = Portfolio(cash=50_000.0, initial_capital=1_000_000.0)
    pf.apply_trade(Trade(20240101, "A", 10.0, 200, "OPEN", cost=5.0))
    pf.apply_trade(Trade(20240101, "B", 20.0, 100, "OPEN", cost=5.0))
    pf2 = Portfolio.from_state(pf.to_state())
    assert pf2.cash == pytest.approx(pf.cash)
    assert pf2.initial_capital == pf.initial_capital
    assert set(pf2.positions) == {"A", "B"}
    assert pf2.positions["A"].quantity == 200


def test_risk_manager_round_trip():
    rm = RiskManager(RiskConfig(stop_loss_pct=0.08), initial_capital=1e6)
    rm.peak_equity = 1_050_000.0
    rm.halted = True
    rm.accumulate_realized(-12_000.0)
    s = rm.to_state()
    assert json.dumps(s)  # JSON-serializable
    rm2 = RiskManager.from_state(s, config=RiskConfig(stop_loss_pct=0.08))
    assert rm2.peak_equity == 1_050_000.0
    assert rm2.halted
    assert rm2._daily_realized == pytest.approx(-12_000.0)
    assert rm2.initial_capital == 1e6


def test_calibration_round_trip():
    cal = CalibrationResult(
        horizon=5, alpha=1.0, factor_names=["ma_dev_20", "momentum_5d"],
        betas={"ma_dev_20": 0.01, "momentum_5d": 0.02},
        ics={"ma_dev_20": 0.1, "momentum_5d": 0.2},
        r_squared=0.3, sample_size=100, n_dates=20,
    )
    cal2 = CalibrationResult.from_state(cal.to_state())
    assert cal2 == cal


def test_pm_round_trip_preserves_account(market, dates, calibration, factors):
    """A run's account state (cash, positions, trades, risk) survives a
    save → load round-trip."""
    bars, _, _ = market
    pm = PortfolioManager(capital=1_000_000.0, n_holdings=5,
                          aggregator=SignalAggregator(calibration=calibration))
    trader = Trader(TradingEngine(factors=factors, calibration=calibration), pm)
    for d in dates[-10:]:
        trader.step(d, bars[d])

    state = pm.to_state()
    eq_before = pm.portfolio.mark_to_market(bars[dates[-1]])

    pm2 = PortfolioManager.from_state(
        state, aggregator=SignalAggregator(calibration=calibration))
    eq_after = pm2.portfolio.mark_to_market(bars[dates[-1]])
    assert eq_after == pytest.approx(eq_before)
    assert pm2._last_date == pm._last_date
    assert len(pm2.trades) == len(pm.trades)
    assert pm2.portfolio.cash == pytest.approx(pm.portfolio.cash)
    for sym, pos in pm.portfolio.positions.items():
        assert pm2.portfolio.positions[sym].quantity == pytest.approx(pos.quantity)
    # Risk state round-trips too.
    assert pm2.risk.halted == pm.risk.halted
    assert pm2.risk.peak_equity == pytest.approx(pm.risk.peak_equity)


def test_trader_round_trip_keeps_pending(market, dates, calibration, factors):
    bars, _, _ = market
    pm = PortfolioManager(capital=1_000_000.0, n_holdings=5,
                          aggregator=SignalAggregator(calibration=calibration))
    trader = Trader(TradingEngine(factors=factors, calibration=calibration), pm)
    for d in dates[-10:]:
        trader.step(d, bars[d])
    assert trader.pending

    state = trader.to_state()
    assert json.dumps(state)  # JSON-serializable

    pm2 = PortfolioManager.from_state(
        pm.to_state(), aggregator=SignalAggregator(calibration=calibration))
    trader2 = Trader.from_state(state,
                                TradingEngine(factors=factors, calibration=calibration),
                                pm2)
    assert len(trader2.pending) == len(trader.pending)
    assert trader2.pending[0].symbol == trader.pending[0].symbol
    assert trader2.pending[0].quantity == trader.pending[0].quantity
    assert len(trader2.order_log) == len(trader.order_log)


def test_calibration_saved_loads_via_state(market, dates, factors, tmp_path):
    """FactorCalibrator.fit → to_state → from_state yields the same model."""
    bars, _, _ = market
    off = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=5, min_samples=20, min_dates=5)
    for d in dates[:-10]:
        cal.collect(d, off.build_cross_section(d, bars[d]), bars)
    result = cal.fit()
    result2 = CalibrationResult.from_state(result.to_state())
    assert result2.betas == result.betas
    assert result2.ics == result.ics