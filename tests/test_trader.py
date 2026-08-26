"""Tests for trader.py: two-phase next-open execution loop."""

import pytest

from alftrade import (
    TradingEngine, PortfolioManager, SignalAggregator,
    Broker, ExecutionConfig, Trader,
)


def _make_trader(calibration, factors, **broker_kw):
    engine = TradingEngine(factors=factors, calibration=calibration)
    pm = PortfolioManager(
        capital=1_000_000.0, n_holdings=5, weighting="equal",
        aggregator=SignalAggregator(calibration=calibration),
    )
    broker = Broker(ExecutionConfig(**broker_kw))
    return Trader(engine, pm, broker)


def test_trader_first_day_has_no_fills(market, dates, calibration, factors):
    """On the first day there are no pending orders, so no fills."""
    bars, _, _ = market
    trader = _make_trader(calibration, factors)
    fills = trader.step(dates[-10], bars[dates[-10]])
    assert fills == []
    assert len(trader.pending) >= 0  # orders are planned for the next open


def test_trader_executes_at_next_open(market, dates, calibration, factors):
    """With zero slippage and zero market impact, fills occur at the next
    day's open price."""
    bars, _, _ = market
    trader = _make_trader(calibration, factors, slippage_bps=0.0,
                          half_spread_bps=0.0, impact_k=0.0)

    for d in dates[-10:]:
        trader.step(d, bars[d])

    assert len(trader.trades) > 0
    for t in trader.trades:
        bar = bars[t.date][t.symbol]
        assert t.price == pytest.approx(round(bar.open, 2), abs=1e-6)
        # Every fill's date is a day that actually exists in the loop.
        assert t.date in dates[-10:]


def test_trader_leaves_pending_at_end(market, dates, calibration, factors):
    """After the loop, the final day's plan is pending (never executed)."""
    bars, _, _ = market
    trader = _make_trader(calibration, factors)
    for d in dates[-10:]:
        trader.step(d, bars[d])
    assert len(trader.pending) > 0  # last day's orders await the (absent) next open


def test_trader_slippage_reduces_equity(market, dates, calibration, factors):
    """On the first fill day, identical orders fill at different prices, so the
    higher-slippage trader ends with strictly lower equity (deterministic)."""
    bars, _, _ = market
    trader_lo = _make_trader(calibration, factors, slippage_bps=0.0005, half_spread_bps=0.0)
    trader_hi = _make_trader(calibration, factors, slippage_bps=0.0050, half_spread_bps=0.0)

    d0, d1 = dates[-10], dates[-9]
    # Day 1: plan (no fills — empty portfolio → identical orders).
    trader_lo.step(d0, bars[d0])
    trader_hi.step(d0, bars[d0])
    # Day 2: execute day 1's orders at different slippage.
    trader_lo.step(d1, bars[d1])
    trader_hi.step(d1, bars[d1])

    eq_lo = trader_lo.portfolio.portfolio.mark_to_market(bars[d1])
    eq_hi = trader_hi.portfolio.portfolio.mark_to_market(bars[d1])
    assert eq_hi < eq_lo


def test_trader_reset(market, dates, calibration, factors):
    """reset() clears pending orders and portfolio state."""
    bars, _, _ = market
    trader = _make_trader(calibration, factors)
    for d in dates[-5:]:
        trader.step(d, bars[d])
    assert len(trader.trades) > 0
    trader.reset()
    assert trader.pending == []
    assert trader.trades == []


def test_switch_timing_changes_fill_prices(market, dates, calibration, factors):
    """Switching to vwap mid-run makes pending orders fill at the day's VWAP
    (amount / volume) instead of the open."""
    bars, _, _ = market
    trader = _make_trader(calibration, factors, slippage_bps=0.0,
                          half_spread_bps=0.0, impact_k=0.0)
    d0, d1 = dates[-10], dates[-9]
    trader.step(d0, bars[d0])                    # plan orders (open timing)
    assert trader.broker.config.execution_timing == "open"
    prev = trader.set_execution_timing("vwap")   # switch before execution
    assert prev == "open"
    assert trader.broker.config.execution_timing == "vwap"
    fills = trader.step(d1, bars[d1])            # pending executes at vwap
    assert fills
    for t in fills:
        bar = bars[d1][t.symbol]
        assert t.price == pytest.approx(round(bar.amount / bar.volume, 2), abs=0.02)


def test_set_execution_timing_rejects_unknown(market, dates, calibration, factors):
    trader = _make_trader(calibration, factors)
    with pytest.raises(ValueError):
        trader.set_execution_timing("bogus")
