"""Tests for broker.py: next-open execution with slippage, volume caps,
price-limit lock, call-auction liquidity, market impact, and the order
journal."""

import pytest

from alftrade.data import DailyBar, compute_limit_prices
from alftrade.portfolio import Trade, CostModel
from alftrade.broker import Broker, ExecutionConfig, OrderRecord


def _bar(sym, open_, close, prev_close, volume=100_000.0, limit_pct=0.10,
         suspended=False):
    up, dn = compute_limit_prices(prev_close, limit_pct)
    return DailyBar(
        date=2, symbol=sym, open=open_, high=max(open_, close), low=min(open_, close),
        close=close, volume=volume, amount=close * volume, prev_close=prev_close,
        limit_up_price=up, limit_down_price=dn, is_suspended=suspended,
    )


def _order(symbol="A", quantity=1000, direction="OPEN"):
    return Trade(date=1, symbol=symbol, price=100.0, quantity=quantity,
                 direction=direction, reason="REBALANCE")


# ── Execution price / slippage ─────────────────────────────────────────

def test_buy_executes_above_open():
    """A buy fills at open × (1 + fixed slippage + half-spread)."""
    broker = Broker(ExecutionConfig(slippage_bps=0.0025, half_spread_bps=0.0005,
                                    impact_k=0.0))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0)
    fills = broker.execute(2, [_order()], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(round(100.0 * (1 + 0.0030), 2))


def test_sell_executes_below_open():
    """A sell fills at open × (1 - fixed slippage - half-spread)."""
    broker = Broker(ExecutionConfig(slippage_bps=0.0025, half_spread_bps=0.0005,
                                    impact_k=0.0))
    bar = _bar("A", open_=100.0, close=99.0, prev_close=101.0)
    fills = broker.execute(2, [_order(quantity=-1000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(round(100.0 * (1 - 0.0030), 2))


def test_market_impact_scales_with_order_size():
    """A 1%-of-day-volume order is 10% of the open-auction liquidity, so it
    pays impact_k × 10 = 20 bps extra (auction impact >> whole-day impact)."""
    broker = Broker(ExecutionConfig(slippage_bps=0.0025, half_spread_bps=0.0005,
                                    impact_k=2.0))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=1_000_000)
    # order = 1% of day volume → 10% of auction volume → 20 bps impact;
    # base 30 bps → 50 bps total.
    fills = broker.execute(2, [_order(quantity=10_000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(round(100.0 * (1 + 0.0050), 2))


def test_cost_computed():
    """The fill carries the transaction cost."""
    broker = Broker(ExecutionConfig())
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=10_000_000)
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar}).fills
    assert fills[0].cost > 0


# ── Call-auction liquidity + participation cap ─────────────────────────

def test_volume_cap_limits_buy():
    """Buys are capped at max_participation × auction volume, rounded to lot.

    auction volume = 0.10 × 5000 = 500; cap = 0.20 × 500 = 100 → 1 lot.
    """
    broker = Broker(ExecutionConfig(max_participation=0.20))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=5000.0)
    fills = broker.execute(2, [_order(quantity=10_000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].quantity == 100


def test_volume_cap_allows_full_fill_when_small():
    """An order within the auction participation cap fills fully."""
    broker = Broker(ExecutionConfig(max_participation=0.20))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=1_000_000)
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].quantity == 1000


def test_large_order_is_partial_with_reason():
    """An oversized order fills partially and is journaled as PARTIAL."""
    broker = Broker(ExecutionConfig(max_participation=0.20))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=5000.0)
    result = broker.execute(2, [_order(quantity=10_000)], {"A": bar})
    rec = result.records[0]
    assert rec.status == "PARTIAL"
    assert "volume cap" in rec.reason
    assert rec.filled_qty == 100


# ── Price-limit lock (checked at the OPEN) ─────────────────────────────

def test_limit_up_open_rejects_buy():
    broker = Broker(ExecutionConfig())
    bar = _bar("A", open_=110.0, close=110.0, prev_close=100.0)  # +10% at open
    assert bar.is_limit_up
    result = broker.execute(2, [_order(quantity=1000)], {"A": bar})
    assert result.fills == []
    assert result.records[0].status == "REJECTED"
    assert "limit-up" in result.records[0].reason


def test_limit_down_open_rejects_sell():
    broker = Broker(ExecutionConfig())
    bar = _bar("A", open_=90.0, close=90.0, prev_close=100.0)  # -10% at open
    assert bar.is_limit_down
    result = broker.execute(2, [_order(quantity=-1000)], {"A": bar})
    assert result.fills == []
    assert result.records[0].status == "REJECTED"
    assert "limit-down" in result.records[0].reason


def test_open_not_locked_allows_fill_even_if_close_locks():
    """The lock uses the OPEN, not the close: a stock that gaps up to its
    limit only after the auction can still be bought at the open."""
    broker = Broker(ExecutionConfig())
    bar = _bar("A", open_=100.0, close=110.0, prev_close=100.0)  # limit-up close
    assert bar.is_limit_up          # close-based flag
    result = broker.execute(2, [_order(quantity=1000)], {"A": bar})
    assert len(result.fills) == 1
    assert result.records[0].status == "FILLED"


def test_limit_lock_configurable():
    """With reject_limit_locked=False, limit-locked fills go through."""
    broker = Broker(ExecutionConfig(reject_limit_locked=False))
    bar = _bar("A", open_=110.0, close=110.0, prev_close=100.0)
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar}).fills
    assert len(fills) == 1


# ── Suspension / missing bar ───────────────────────────────────────────

def test_suspended_symbol_not_filled():
    broker = Broker(ExecutionConfig())
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, suspended=True)
    result = broker.execute(2, [_order()], {"A": bar})
    assert result.fills == []
    assert result.records[0].status == "SKIPPED"
    assert "suspended" in result.records[0].reason


def test_missing_bar_not_filled():
    broker = Broker(ExecutionConfig())
    result = broker.execute(2, [_order(symbol="Z")], {"A": _bar("A", 100, 101, 99)})
    assert result.fills == []
    assert result.records[0].status == "SKIPPED"
    assert "no bar" in result.records[0].reason


# ── Order journal ──────────────────────────────────────────────────────

def test_journal_has_one_record_per_order():
    broker = Broker(ExecutionConfig())
    bars = {"A": _bar("A", 100.0, 101.0, 99.0, volume=1_000_000),
            "B": _bar("B", 90.0, 90.0, 100.0)}   # B limit-down at open
    result = broker.execute(2, [_order("A"), _order("B", quantity=-500)], bars)
    assert len(result.records) == 2
    assert result.records[0].status == "FILLED"
    assert result.records[1].status == "REJECTED"
    assert isinstance(result.records[0], OrderRecord)
    assert result.records[0].filled_qty == 1000
    assert result.records[1].filled_qty == 0


# ── Execution timing: close_auction ────────────────────────────────────

def test_close_auction_fills_at_close():
    """close_auction fills at the day's close price (carrying the full move)."""
    broker = Broker(ExecutionConfig(execution_timing="close_auction", impact_k=0.0))
    bar = _bar("A", open_=100.0, close=102.0, prev_close=99.0)
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(round(102.0 * (1 + 0.0030), 2))


def test_close_auction_lock_uses_close():
    """close_auction checks the limit against the CLOSE, not the open: a stock
    that locks up intraday but closes flat can still be bought at close."""
    broker = Broker(ExecutionConfig(execution_timing="close_auction"))
    # open 100 (not locked), close 110 = limit-up close.
    bar = _bar("A", open_=100.0, close=110.0, prev_close=100.0)
    assert bar.is_limit_up
    result = broker.execute(2, [_order(quantity=1000)], {"A": bar})
    assert result.fills == []
    assert result.records[0].status == "REJECTED"
    assert "limit-up" in result.records[0].reason
    # inverse: open at limit but close below → close_auction fills.
    bar2 = _bar("A", open_=110.0, close=108.0, prev_close=100.0)
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar2}).fills
    assert len(fills) == 1


# ── Execution timing: vwap ─────────────────────────────────────────────

def test_vwap_fills_at_amount_over_volume():
    """vwap fills at the day's VWAP = amount / volume."""
    broker = Broker(ExecutionConfig(execution_timing="vwap", impact_k=0.0))
    bar = _bar("A", open_=100.0, close=102.0, prev_close=99.0, volume=100_000.0)
    # amount = 10,150,000 → VWAP = 101.50
    bar = DailyBar(
        date=2, symbol="A", open=100.0, high=103.0, low=99.0, close=102.0,
        volume=100_000.0, amount=10_150_000.0, prev_close=99.0,
        limit_up_price=108.9, limit_down_price=89.1,
    )
    fills = broker.execute(2, [_order(quantity=1000)], {"A": bar}).fills
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(round(101.50 * (1 + 0.0030), 2))


def test_vwap_uses_full_day_volume_for_participation():
    """vwap trades against the whole day's volume (spread out), so the same
    order fills fully where an auction order would only partially fill."""
    broker = Broker(ExecutionConfig(execution_timing="vwap", max_participation=0.20))
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=5000.0)
    # open-timing cap = 0.20 × 500 = 100; vwap cap = 0.20 × 5000 = 1000.
    fills = broker.execute(2, [_order(quantity=10_000)], {"A": bar}).fills
    assert fills[0].quantity == 1000


def test_vwap_impact_uses_full_day_denominator():
    """The same order pays less *impact* under vwap (whole-day denominator).
    Compare implied impact (fill/reference − 1), not raw prices."""
    bar = _bar("A", open_=100.0, close=101.0, prev_close=99.0, volume=1_000_000)
    broker_open = Broker(ExecutionConfig(impact_k=2.0))
    broker_vwap = Broker(ExecutionConfig(impact_k=2.0, execution_timing="vwap"))
    f_open = broker_open.execute(2, [_order(quantity=10_000)], {"A": bar}).fills[0]
    f_vwap = broker_vwap.execute(2, [_order(quantity=10_000)], {"A": bar}).fills[0]
    imp_open = f_open.price / bar.open - 1
    imp_vwap = f_vwap.price / (bar.amount / bar.volume) - 1
    assert imp_vwap < imp_open


def test_vwap_lock_checks_open():
    """vwap uses the OPEN for the lock: if the open is not locked there was
    intraday opportunity even when the close locks."""
    broker = Broker(ExecutionConfig(execution_timing="vwap"))
    bar = _bar("A", open_=100.0, close=110.0, prev_close=100.0)  # limit-up close
    assert bar.is_limit_up
    result = broker.execute(2, [_order(quantity=1000)], {"A": bar})
    assert len(result.fills) == 1
    # open at limit → no intraday chance → rejected.
    bar2 = _bar("A", open_=110.0, close=110.0, prev_close=100.0)
    result2 = broker.execute(2, [_order(quantity=1000)], {"A": bar2})
    assert result2.fills == []
    assert result2.records[0].status == "REJECTED"


def test_unknown_timing_raises():
    broker = Broker(ExecutionConfig(execution_timing="bogus"))
    with pytest.raises(ValueError):
        broker.execute(2, [_order()], {"A": _bar("A", 100, 101, 99)})
