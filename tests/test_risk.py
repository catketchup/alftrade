"""Tests for risk.py: all A-share pre-trade and post-bar risk rules.

Pre-trade:  suspension, limit-lock, T+1, board lot, concentration,
            notional cap, order frequency, circuit breaker.
Post-bar:   stop-loss, take-profit, max drawdown, daily loss limit.
"""

import pytest
from alftrade.data import DailyBar, compute_limit_prices
from alftrade.portfolio import (
    Position, Trade, Portfolio, CostModel, BOARD_LOT,
)
from alftrade.risk import RiskManager, RiskConfig


def _bar(sym, close, prev_close=None, suspended=False, limit_pct=0.10):
    pc = prev_close if prev_close is not None else close
    up, dn = compute_limit_prices(pc, limit_pct)
    return DailyBar(
        date=1, symbol=sym, open=pc, high=close, low=close, close=close,
        volume=100_000.0, amount=close * 100_000.0, prev_close=pc,
        limit_up_price=up, limit_down_price=dn, is_suspended=suspended,
    )


def _cfg(**overrides):
    """A permissive config; tests enable one rule at a time via overrides."""
    base = dict(
        enforce_t1=True, enforce_lot=True, reject_limit_locked=True,
        reject_suspended=True, max_weight_per_name=None,
        max_notional_per_trade=None, max_trades_per_day=None,
        stop_loss_pct=None, take_profit_pct=None,
        max_drawdown_pct=None, daily_loss_limit_pct=None,
    )
    base.update(overrides)
    return RiskConfig(**base)


# ── Pre-trade: suspension ──────────────────────────────────────────────

def test_suspension_rejects_both_sides():
    """Suspended symbols reject both buys and sells."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 100.0, suspended=True)
    pf = Portfolio(cash=1_000_000.0)
    buy = Trade(1, "A", 100.0, 100, "OPEN")
    assert rm.check_pre_trade({"A": bar}, buy, pf) is None
    # Pre-load a position then try to sell.
    pf.positions["A"] = Position(quantity=100, avg_entry_price=100.0)
    sell = Trade(1, "A", 100.0, -100, "CLOSE")
    assert rm.check_pre_trade({"A": bar}, sell, pf) is None


# ── Pre-trade: price-limit lock ────────────────────────────────────────

def test_limit_up_rejects_buys():
    """Buys at limit-up are rejected (queue is all buyers)."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 110.0, prev_close=100.0)  # at +10% limit
    assert bar.is_limit_up
    pf = Portfolio(cash=1_000_000.0)
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 110.0, 100, "OPEN"), pf) is None


def test_limit_down_rejects_sells():
    """Sells at limit-down are rejected (queue is all sellers)."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 90.0, prev_close=100.0)  # at -10% limit
    assert bar.is_limit_down
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=100, avg_entry_price=100.0)})
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 90.0, -100, "CLOSE"), pf) is None


def test_limit_up_allows_sells():
    """Sells are still allowed at limit-up (you can sell into the bid queue)."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 110.0, prev_close=100.0)
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=100, avg_entry_price=100.0)})
    pf.positions["A"].t1_locked = 0  # make sellable
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 110.0, -100, "CLOSE"), pf)
    assert result is not None


# ── Pre-trade: T+1 ─────────────────────────────────────────────────────

def test_t1_caps_sell_to_available():
    """A sell exceeding available_qty is shrunk, not rejected."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=200, avg_entry_price=100.0)})
    # 100 shares locked (bought today), 100 available.
    pf.positions["A"].t1_locked = 100
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, -200, "CLOSE"), pf)
    assert result is not None
    assert result.quantity == -100   # shrunk to available_qty


def test_t1_rejects_when_nothing_available():
    """A sell when available_qty = 0 is rejected."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=100, avg_entry_price=100.0)})
    pf.positions["A"].t1_locked = 100  # all locked
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, -100, "CLOSE"), pf) is None


def test_t1_disabled_allows_full_sell():
    """With enforce_t1=False, sells are not capped by available_qty."""
    rm = RiskManager(_cfg(enforce_t1=False))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=100, avg_entry_price=100.0)})
    pf.positions["A"].t1_locked = 100
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, -100, "CLOSE"), pf)
    assert result is not None
    assert result.quantity == -100


# ── Pre-trade: board lot ───────────────────────────────────────────────

def test_lot_rounds_buys_down():
    """Buys that aren't a multiple of 100 are rounded down."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 250, "OPEN"), pf)
    assert result is not None
    assert result.quantity == 200


def test_lot_rejects_sublot_buy():
    """A buy smaller than one board lot is rejected."""
    rm = RiskManager(_cfg())
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 50, "OPEN"), pf) is None


# ── Pre-trade: notional cap ────────────────────────────────────────────

def test_notional_cap_shrinks_buy():
    """A buy exceeding max_notional is shrunk to the cap (and lot-rounded)."""
    rm = RiskManager(_cfg(max_notional_per_trade=50_000.0))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 1000, "OPEN"), pf)
    assert result is not None
    # 50,000 / 100 = 500 shares, already a multiple of 100.
    assert result.quantity == 500


def test_notional_cap_passes_small_order():
    """Orders within the cap pass unchanged."""
    rm = RiskManager(_cfg(max_notional_per_trade=50_000.0))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 200, "OPEN"), pf)
    assert result is not None
    assert result.quantity == 200


# ── Pre-trade: single-name concentration ───────────────────────────────

def test_concentration_shrinks_buy_to_cap():
    """A buy that would breach max_weight_per_name is shrunk."""
    rm = RiskManager(_cfg(max_weight_per_name=0.10))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    # 10% of 1M equity = 100,000 → 1,000 shares at 100. Asking for 2,000.
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 2000, "OPEN"), pf)
    assert result is not None
    assert result.quantity == 1000


def test_concentration_rejects_when_already_at_cap():
    """If the position is already at the cap, further buys are rejected."""
    rm = RiskManager(_cfg(max_weight_per_name=0.10))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=900_000.0,
                   positions={"A": Position(quantity=1000, avg_entry_price=100.0)})
    # equity = 900,000 + 100,000 = 1,000,000; cap = 100,000; already at cap.
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 100, "INCREASE"), pf) is None


# ── Pre-trade: order frequency ─────────────────────────────────────────

def test_order_frequency_rejects_excess():
    """Too many trades in one day rejects further trades."""
    rm = RiskManager(_cfg(max_trades_per_day=3))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    for _ in range(3):
        rm.record_trade(1, "A")
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 100, "OPEN"), pf) is None


def test_order_frequency_resets_next_day():
    """The per-day counter resets when the date advances."""
    rm = RiskManager(_cfg(max_trades_per_day=2))
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    rm.record_trade(1, "A")
    rm.record_trade(1, "A")
    # Day 2: counter resets.
    result = rm.check_pre_trade({"A": bar}, Trade(2, "A", 100.0, 100, "OPEN"), pf)
    assert result is not None


# ── Pre-trade: circuit breaker ─────────────────────────────────────────

def test_circuit_breaker_blocks_opens():
    """When halted, new OPEN buys are rejected."""
    rm = RiskManager(_cfg())
    rm.halted = True
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=1_000_000.0)
    assert rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, 100, "OPEN"), pf) is None


def test_circuit_breaker_allows_closes():
    """When halted, CLOSE sells still pass (must be able to exit)."""
    rm = RiskManager(_cfg())
    rm.halted = True
    bar = _bar("A", 100.0)
    pf = Portfolio(cash=0.0, positions={"A": Position(quantity=100, avg_entry_price=100.0)})
    result = rm.check_pre_trade({"A": bar}, Trade(1, "A", 100.0, -100, "CLOSE"), pf)
    assert result is not None


# ── Post-bar: stop loss ────────────────────────────────────────────────

def test_stop_loss_forces_close():
    """A position losing > stop_loss_pct of cost is force-closed."""
    rm = RiskManager(_cfg(stop_loss_pct=0.05))
    # Entry 100, now 94 → loss = 6% > 5%.
    pf = Portfolio(cash=0.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 94.0, prev_close=100.0)
    forced = rm.check_post_bar(1, {"A": bar}, pf)
    assert len(forced) == 1
    assert "STOP_LOSS" in forced[0].reason
    assert forced[0].quantity == -100


def test_stop_loss_no_trigger_inside_band():
    """A small loss within the band does not trigger."""
    rm = RiskManager(_cfg(stop_loss_pct=0.05))
    pf = Portfolio(cash=0.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 97.0, prev_close=100.0)  # 3% loss < 5%
    assert rm.check_post_bar(1, {"A": bar}, pf) == []


# ── Post-bar: take profit ──────────────────────────────────────────────

def test_take_profit_forces_close():
    """A position gaining > take_profit_pct of cost is force-closed."""
    rm = RiskManager(_cfg(take_profit_pct=0.10))
    pf = Portfolio(cash=0.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 112.0, prev_close=100.0)  # 12% gain > 10%
    forced = rm.check_post_bar(1, {"A": bar}, pf)
    assert len(forced) == 1
    assert "TAKE_PROFIT" in forced[0].reason


# ── Post-bar: max drawdown ─────────────────────────────────────────────

def test_max_drawdown_halts_and_liquidates():
    """A drawdown beyond the threshold halts and closes everything."""
    rm = RiskManager(_cfg(max_drawdown_pct=0.20))
    rm.peak_equity = 1_000_000.0
    # Current equity 780,000 → drawdown = 22% > 20%.
    pf = Portfolio(cash=780_000.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 100.0, prev_close=100.0)  # position mv = 10,000 → equity 790,000
    # Actually equity = 780k cash + 100*100 = 790k. drawdown = (1M - 790k)/1M = 21%.
    forced = rm.check_post_bar(1, {"A": bar}, pf)
    assert rm.halted
    assert any("MAX_DRAWDOWN" in t.reason for t in forced)


def test_max_drawdown_no_trigger_inside_band():
    """A modest drawdown does not trigger."""
    rm = RiskManager(_cfg(max_drawdown_pct=0.20))
    rm.peak_equity = 1_000_000.0
    pf = Portfolio(cash=900_000.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 100.0, prev_close=100.0)  # equity 910k → dd 9% < 20%
    rm.check_post_bar(1, {"A": bar}, pf)
    assert not rm.halted


# ── Post-bar: daily loss limit ─────────────────────────────────────────

def test_daily_loss_limit_halts():
    """Daily realized loss beyond the limit halts the system."""
    rm = RiskManager(_cfg(daily_loss_limit_pct=0.05), initial_capital=1_000_000.0)
    rm._current_date = 1
    rm._daily_realized = -60_000.0   # -6% < -5%
    pf = Portfolio(cash=1_000_000.0, positions={
        "A": Position(quantity=100, avg_entry_price=100.0, t1_locked=0.0)})
    bar = _bar("A", 100.0, prev_close=100.0)
    forced = rm.check_post_bar(1, {"A": bar}, pf)
    assert rm.halted
    assert any("DAILY_LOSS_LIMIT" in t.reason for t in forced)


# ── reset ──────────────────────────────────────────────────────────────

def test_reset_clears_state():
    """reset() clears halt, peak equity, and day counters."""
    rm = RiskManager(_cfg())
    rm.halted = True
    rm.peak_equity = 1_000_000.0
    rm._trades_today = 5
    rm._daily_realized = -100.0
    rm.reset()
    assert rm.halted is False
    assert rm.peak_equity == 0.0
    assert rm._trades_today == 0
    assert rm._daily_realized == 0.0
