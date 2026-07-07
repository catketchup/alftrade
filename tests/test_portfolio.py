"""Tests for portfolio.py: Position, Trade, CostModel, Portfolio,
SignalAggregator, and PortfolioManager.

Covers:
  - Position.apply_trade: open / increase / reduce / close (long-only, T+1)
  - Position.mark_to_market
  - CostModel: commission + stamp duty + min commission
  - Portfolio: cash impact, T+1 rollover, mark-to-market equity
  - SignalAggregator: weighted voting, empty, non-directional, "model" fallback
  - PortfolioManager: open basket, rebalance, drop names, cap by cash
"""

import pytest
from alftrade.data import DailyBar, compute_limit_prices
from alftrade.engine import Signal, TradingEngine
from alftrade.portfolio import (
    Position, Trade, Portfolio, CostModel, CompositeSignal,
    SignalAggregator, PortfolioManager, BOARD_LOT, round_lot_down,
)


def _bar(sym, close, prev_close=None, suspended=False):
    pc = prev_close if prev_close is not None else close
    up, dn = compute_limit_prices(pc, 0.10)
    return DailyBar(
        date=1, symbol=sym, open=pc, high=close, low=close, close=close,
        volume=10000.0, amount=close * 10000.0, prev_close=pc,
        limit_up_price=up, limit_down_price=dn, is_suspended=suspended,
    )


# ── round_lot_down ─────────────────────────────────────────────────────

def test_round_lot_down():
    """Buy quantities round down to the nearest 100 shares."""
    assert round_lot_down(250) == 200
    assert round_lot_down(99) == 0
    assert round_lot_down(100) == 100
    assert round_lot_down(0) == 0


# ── CostModel ──────────────────────────────────────────────────────────

def test_cost_model_buy_no_stamp_duty():
    """Buys pay commission + transfer fee but no stamp duty."""
    cm = CostModel()
    cost = cm.cost("BUY", price=10.0, quantity=1000)
    notional = 10.0 * 1000
    expected = max(notional * 0.00025, 5.0) + notional * 0.00002
    assert cost == pytest.approx(expected)


def test_cost_model_sell_includes_stamp_duty():
    """Sells pay commission + transfer fee + stamp duty (10 bps)."""
    cm = CostModel()
    cost = cm.cost("SELL", price=10.0, quantity=1000)
    notional = 10.0 * 1000
    expected = max(notional * 0.00025, 5.0) + notional * 0.00002 + notional * 0.001
    assert cost == pytest.approx(expected)


def test_cost_model_min_commission():
    """Small trades pay the 5-yuan minimum commission, not the bps amount."""
    cm = CostModel()
    # 100 shares × 10 yuan = 1000 notional → 0.25 yuan commission < 5 min.
    cost = cm.cost("BUY", price=10.0, quantity=100)
    expected = 5.0 + 1000 * 0.00002
    assert cost == pytest.approx(expected)


def test_cost_model_zero_quantity():
    """Zero-quantity trades cost nothing."""
    cm = CostModel()
    assert cm.cost("BUY", 10.0, 0) == 0.0
    assert cm.cost("SELL", 10.0, 0) == 0.0


# ── Position.apply_trade ───────────────────────────────────────────────

def test_open_long():
    """Opening a long sets qty, avg entry, and T+1 lock."""
    pos = Position()
    t = Trade(1, "A", 100.0, 100, "OPEN", cost=2.5)
    pos.apply_trade(t)
    assert pos.quantity == 100
    assert pos.avg_entry_price == 100.0
    assert pos.t1_locked == 100      # bought today → locked
    assert pos.available_qty == 0    # nothing sellable today
    assert pos.realized_pnl == pytest.approx(-2.5)  # cost deducted


def test_increase_long_weighted_avg():
    """Increasing a long updates weighted avg entry and adds to T1 lock.

    A rollover_t1() between the two buys models the new-day boundary: the
    first buy becomes sellable, and only the second (same-day) buy stays
    locked under T+1.
    """
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN"))
    pos.rollover_t1()                  # new day → first buy unlocks
    pos.apply_trade(Trade(2, "A", 110.0, 100, "INCREASE"))
    assert pos.quantity == 200
    assert pos.avg_entry_price == pytest.approx(105.0)
    assert pos.t1_locked == 100        # only the second buy is locked today
    assert pos.available_qty == 100    # first buy is now sellable


def test_t1_rollover_frees_shares():
    """After rollover_t1, locked shares become sellable."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN"))
    assert pos.available_qty == 0
    pos.rollover_t1()
    assert pos.t1_locked == 0
    assert pos.available_qty == 100


def test_sell_blocked_by_t1():
    """Selling more than available_qty is capped (no short created)."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN", cost=2.5))   # locked
    # Try to sell 100 today → capped to available_qty = 0 → no-op.
    pos.apply_trade(Trade(1, "A", 100.0, -100, "CLOSE", cost=10.0))
    assert pos.quantity == 100          # unchanged
    assert pos.realized_pnl == -2.5     # only the buy's cost was deducted (sell was a no-op)


def test_sell_after_rollover_realizes_pnl():
    """After T+1 rollover, a sell realizes PnL and deducts cost."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN", cost=2.5))
    pos.rollover_t1()
    pos.apply_trade(Trade(2, "A", 105.0, -100, "CLOSE", cost=10.0))
    assert pos.is_flat
    # PnL = (105 - 100) * 100 - buy_cost - sell_cost = 500 - 2.5 - 10 = 487.5
    assert pos.realized_pnl == pytest.approx(487.5)


def test_partial_close_keeps_avg_entry():
    """A partial close realizes PnL on the closed portion only."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 200, "OPEN"))
    pos.rollover_t1()
    pos.apply_trade(Trade(2, "A", 110.0, -50, "REDUCE"))
    assert pos.quantity == 150
    assert pos.avg_entry_price == pytest.approx(100.0)
    assert pos.realized_pnl == pytest.approx(10.0 * 50)  # (110-100)*50


def test_sell_cannot_create_short():
    """Even a sell larger than the position stops at zero (long-only)."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN"))
    pos.rollover_t1()
    pos.apply_trade(Trade(2, "A", 105.0, -500, "CLOSE"))  # try to sell 500
    assert pos.quantity == 0   # flattened, not short
    assert pos.is_flat


# ── Position.mark_to_market ────────────────────────────────────────────

def test_mark_to_market_long():
    """Long unrealized PnL = (current - entry) * qty."""
    pos = Position()
    pos.apply_trade(Trade(1, "A", 100.0, 100, "OPEN"))
    pos.mark_to_market(110.0)
    assert pos.unrealized_pnl == pytest.approx(10.0 * 100)


def test_mark_to_market_flat_is_zero():
    """A flat position has zero unrealized PnL."""
    pos = Position()
    pos.mark_to_market(999.0)
    assert pos.unrealized_pnl == 0.0


# ── Portfolio ──────────────────────────────────────────────────────────

def test_portfolio_cash_decreases_on_buy():
    """A buy reduces cash by notional + cost."""
    pf = Portfolio(cash=1_000_000.0)
    pf.apply_trade(Trade(1, "A", 50.0, 200, "OPEN", cost=5.0))
    # cash = 1,000,000 - 200*50 - 5 = 989,995
    assert pf.cash == pytest.approx(1_000_000 - 200 * 50 - 5)


def test_portfolio_cash_increases_on_sell():
    """A sell increases cash by notional - cost."""
    pf = Portfolio(cash=1_000_000.0)
    pf.apply_trade(Trade(1, "A", 50.0, 200, "OPEN", cost=5.0))
    pf.rollover_t1()
    pf.apply_trade(Trade(2, "A", 55.0, -200, "CLOSE", cost=6.0))
    # After buy: 989,995. After sell: + 200*55 - 6 = +10,994 → 1,000,989
    assert pf.cash == pytest.approx(989_995 + 200 * 55 - 6)


def test_portfolio_rollover_t1_clears_all():
    """rollover_t1 clears the lock on every position."""
    pf = Portfolio(cash=1_000_000.0)
    pf.apply_trade(Trade(1, "A", 50.0, 100, "OPEN"))
    pf.apply_trade(Trade(1, "B", 30.0, 100, "OPEN"))
    assert pf.positions["A"].t1_locked == 100
    assert pf.positions["B"].t1_locked == 100
    pf.rollover_t1()
    assert pf.positions["A"].t1_locked == 0
    assert pf.positions["B"].t1_locked == 0


def test_portfolio_mark_to_market_equity():
    """Equity = cash + Σ qty × close after mark-to-market."""
    pf = Portfolio(cash=1_000_000.0)
    pf.apply_trade(Trade(1, "A", 100.0, 100, "OPEN", cost=5.0))
    bars = {"A": _bar("A", 110.0, prev_close=100.0)}
    equity = pf.mark_to_market(bars)
    # cash = 1,000,000 - 100*100 - 5 = 989,995; position value = 100*110 = 11,000
    assert equity == pytest.approx(989_995 + 100 * 110)


# ── SignalAggregator ───────────────────────────────────────────────────

def test_aggregate_empty():
    """Empty signals → zero composite."""
    agg = SignalAggregator()
    cs = agg.aggregate("A", [])
    assert cs.score == 0.0
    assert cs.contributing == 0
    assert cs.direction == 0


def test_aggregate_single_long():
    """One bullish signal → direction=+1, score = strength × weight."""
    agg = SignalAggregator()
    cs = agg.aggregate("A", [Signal("A", "momentum_5d", 1, 0.5)])
    assert cs.direction == 1
    assert cs.score == pytest.approx(0.5)
    assert cs.contributing == 1


def test_aggregate_skips_flat_signals():
    """direction=0 signals still contribute 0 to score (and to count)."""
    agg = SignalAggregator()
    cs = agg.aggregate("A", [Signal("A", "wide_spread", 0, 5.0)])
    # direction 0 → contributes 0; score 0 → composite direction 0
    assert cs.score == 0.0
    assert cs.direction == 0


def test_aggregate_weighted_override():
    """Manual weights override calibration IC."""
    agg = SignalAggregator(weights={"x": 2.0, "y": 1.0})
    cs = agg.aggregate("A", [
        Signal("A", "x", 1, 0.3),   # 0.6
        Signal("A", "y", 1, 0.5),   # 0.5
    ])
    assert cs.score == pytest.approx(1.1)
    assert cs.direction == 1


def test_aggregate_model_signal_falls_back_to_unit_weight(calibration):
    """The "model" signal name (not in calibration.ics) gets weight 1.0.

    This is the calibrated-mode path: the engine emits one "model" signal
    whose strength is already the predicted return, so it must NOT be
    zeroed out by an IC lookup miss.
    """
    agg = SignalAggregator(calibration=calibration)
    cs = agg.aggregate("A", [Signal("A", "model", 1, 0.0123)])
    assert cs.score == pytest.approx(0.0123)
    assert cs.direction == 1


def test_aggregate_all_returns_per_symbol():
    """aggregate_all returns one CompositeSignal per symbol with signals."""
    agg = SignalAggregator()
    out = agg.aggregate_all({
        "A": [Signal("A", "x", 1, 0.5)],
        "B": [Signal("B", "x", 0, 0.1)],
    })
    assert set(out.keys()) == {"A", "B"}
    assert out["A"].direction == 1
    assert out["B"].direction == 0


# ── PortfolioManager ───────────────────────────────────────────────────

def _pm(capital=1_000_000.0, n_holdings=3, **kw):
    """Build a PortfolioManager with a permissive risk config for unit tests."""
    from alftrade.risk import RiskManager, RiskConfig
    cfg = RiskConfig(
        enforce_t1=True, enforce_lot=True, reject_limit_locked=True,
        reject_suspended=True, max_weight_per_name=0.5,
        max_notional_per_trade=None, max_trades_per_day=None,
        stop_loss_pct=None, take_profit_pct=None,
        max_drawdown_pct=None, daily_loss_limit_pct=None,
    )
    return PortfolioManager(
        capital=capital, n_holdings=n_holdings,
        aggregator=SignalAggregator(),
        risk_manager=RiskManager(cfg),
        **kw,
    )


def test_pm_opens_basket_on_strong_signals():
    """PM opens positions in the top-N scored stocks."""
    pm = _pm(n_holdings=3)
    bars = {f"S{i}": _bar(f"S{i}", 10.0) for i in range(5)}
    signals = {f"S{i}": [Signal(f"S{i}", "model", 1, 0.01 + i * 0.001)]
               for i in range(5)}
    trades = pm.decide(1, bars, signals)
    # 3 buys expected (top-3 by score).
    buys = [t for t in trades if t.quantity > 0]
    assert len(buys) == 3
    # The top-3 scored stocks are S2, S3, S4.
    opened = {t.symbol for t in buys}
    assert opened == {"S2", "S3", "S4"}


def test_pm_no_action_on_no_signals():
    """No signals → no trades, portfolio stays flat."""
    pm = _pm()
    bars = {f"S{i}": _bar(f"S{i}", 10.0) for i in range(3)}
    trades = pm.decide(1, bars, {})
    assert trades == []
    assert all(p.is_flat for p in pm.portfolio.positions.values())


def test_pm_rebalance_drops_weakest_name():
    """When a stronger name appears, the weakest holding is sold (T+1 first)."""
    pm = _pm(n_holdings=2)
    bars = {f"S{i}": _bar(f"S{i}", 10.0) for i in range(4)}
    # Day 1: long S2, S3 (top 2 of S0..S3 by score).
    pm.decide(1, bars, {f"S{i}": [Signal(f"S{i}", "model", 1, i * 0.01)]
                        for i in range(4)})
    assert pm.portfolio.positions["S3"].quantity > 0
    assert pm.portfolio.positions["S2"].quantity > 0
    # Day 2: roll T+1, then scores change so S3 should be dropped for... no
    # new stock: S2, S3 are still top-2. Instead introduce S0 as the new top.
    pm.decide(2, bars, {f"S{i}": [Signal(f"S{i}", "model", 1, (3 - i) * 0.01)]
                        for i in range(4)})
    # After day 2, top-2 by score are S0 (0.03) and S1 (0.02). S2, S3 should
    # have been reduced/closed. At least one sell should have occurred.
    sells = [t for t in pm.trades if t.quantity < 0 and t.date == 2]
    assert len(sells) >= 1


def test_pm_buys_are_board_lot_multiples():
    """Every buy quantity is a multiple of BOARD_LOT (100)."""
    pm = _pm(n_holdings=3)
    bars = {f"S{i}": _bar(f"S{i}", 33.33) for i in range(5)}  # awkward price
    signals = {f"S{i}": [Signal(f"S{i}", "model", 1, 0.02)]
               for i in range(5)}
    pm.decide(1, bars, signals)
    for t in pm.trades:
        if t.quantity > 0:
            assert int(t.quantity) % BOARD_LOT == 0


def test_pm_long_only_never_short():
    """No position ever goes negative across a multi-day run."""
    pm = _pm(n_holdings=3)
    bars = {f"S{i}": _bar(f"S{i}", 10.0 + i) for i in range(5)}
    for day in range(1, 5):
        # Flip scores each day to force churn.
        scores = {f"S{i}": (0.02 if (i + day) % 2 == 0 else 0.001) for i in range(5)}
        pm.decide(day, bars,
                  {f"S{i}": [Signal(f"S{i}", "model", 1, scores[f"S{i}"])]
                   for i in range(5)})
    for p in pm.portfolio.positions.values():
        assert p.quantity >= 0


def test_pm_t1_blocks_same_day_sell():
    """A name bought today cannot be sold today even if the signal flips."""
    pm = _pm(n_holdings=2)
    bars = {f"S{i}": _bar(f"S{i}", 10.0) for i in range(4)}
    # Day 1: long S2, S3.
    pm.decide(1, bars, {f"S{i}": [Signal(f"S{i}", "model", 1, i * 0.01)]
                        for i in range(4)})
    q_after_buy = {s: pm.portfolio.positions[s].quantity for s in ("S2", "S3")}
    # Day 2 with scores that drop S2/S3 to 0 — but T+1 has rolled by now
    # (rollover happens at start of decide), so they CAN be sold on day 2.
    # The T+1 block is within day 1: feed a *second* decide on the same date.
    pm.decide(1, bars, {f"S{i}": [Signal(f"S{i}", "model", 0, 0.0)]
                        for i in range(4)})
    # Same-day sell blocked: quantities unchanged from after the first buy.
    for s in ("S2", "S3"):
        assert pm.portfolio.positions[s].quantity == q_after_buy[s]


def test_pm_reset():
    """reset() clears positions, trades, and risk state."""
    pm = _pm()
    bars = {f"S{i}": _bar(f"S{i}", 10.0) for i in range(3)}
    pm.decide(1, bars, {f"S{i}": [Signal(f"S{i}", "model", 1, 0.02)]
                        for i in range(3)})
    assert len(pm.trades) > 0
    pm.reset()
    assert pm.portfolio.cash == pm.capital
    assert pm.trades == []
    assert all(p.is_flat for p in pm.portfolio.positions.values())
