"""
risk.py - Risk management layer for the A-share daily cross-sectional system.

Sits between the PortfolioManager's rebalance decisions and the portfolio's
``apply_trade``, exactly like hftrade's ``RiskManager`` sat between
``PositionManager`` and ``Position``. Two entry points:

  1. ``check_pre_trade(bars, trade, portfolio)`` — gate before a proposed
     trade executes. May reject, shrink, or pass through. Implements the
     A-share-specific rules that don't exist in hftrade:
        - Suspension: reject all trades on halted symbols.
        - Price-limit lock: reject buys at limit-up, sells at limit-down.
        - T+1 settlement: shrink sells to ``available_qty``.
        - Board lot: reject/shrink buys that aren't a multiple of 100.
        - Single-name concentration: shrink buys that would breach
          ``max_weight_per_name``.
        - Notional cap: shrink any trade exceeding ``max_notional_per_trade``.
        - Order frequency: reject once ``max_trades_per_day`` is hit.
        - Circuit breaker: reject new opens when the portfolio is halted.

  2. ``check_post_bar(date, bars, portfolio)`` — monitor after the rebalance.
        - Per-position stop-loss / take-profit (pct of position cost).
        - Portfolio max drawdown from peak equity.
        - Daily realized-loss limit.
      Returns a list of forced-close trades (possibly empty), because a
      portfolio can have many positions triggering simultaneously — unlike
      hftrade's single-instrument monitor which returned one.

Data flow:
    PortfolioManager.decide()
        → proposed trades
        → RiskManager.check_pre_trade()    ← gate (reject / shrink / pass)
        → Portfolio.apply_trade()
        → RiskManager.check_post_bar()     ← monitor (force closes)
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from alftrade.data import DailyBar
from alftrade.portfolio import BOARD_LOT, round_lot_down, Trade

if TYPE_CHECKING:
    from alftrade.portfolio import Portfolio, Position


@dataclass
class RiskConfig:
    """Configuration for all risk rules. Set any to None to disable it.

    Defaults are pedagogical — realistic but not aggressive.
    """
    # ── Pre-trade: A-share structural rules ──
    enforce_t1: bool = True                    # Block same-day sells of today's buys
    enforce_lot: bool = True                   # Buys must be a multiple of BOARD_LOT
    reject_limit_locked: bool = True           # Reject buys at limit-up, sells at limit-down
    reject_suspended: bool = True              # Reject all trades on halted symbols

    # ── Pre-trade: portfolio-level caps ──
    max_weight_per_name: Optional[float] = 0.25        # Max portfolio weight per stock
    max_notional_per_trade: Optional[float] = 500_000.0  # Max yuan per single trade
    max_trades_per_day: Optional[int] = 50          # Max total trades per date

    # ── Post-bar: per-position monitors ──
    stop_loss_pct: Optional[float] = 0.10       # Force close if loss > 10% of cost
    take_profit_pct: Optional[float] = None     # Disabled by default (let winners run)

    # ── Post-bar: portfolio-level monitors ──
    max_drawdown_pct: Optional[float] = 0.20    # Halt if equity drops 20% from peak
    daily_loss_limit_pct: Optional[float] = 0.05  # Halt if daily realized loss > 5% of capital


class RiskManager:
    """Pre-trade gate + post-bar monitor for the A-share daily portfolio.

    Holds running state (peak equity, per-date trade counts, halt flag, daily
    realized PnL accumulator) and exposes the two methods called by
    ``PortfolioManager``.

    Args:
        config: RiskConfig with thresholds. Defaults are pedagogical.
        initial_capital: Used as the denominator for ``daily_loss_limit_pct``.
            If None, inferred from the portfolio's ``initial_capital`` the
            first time ``check_post_bar`` runs.
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_capital: Optional[float] = None,
    ):
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.peak_equity: float = 0.0
        self.halted: bool = False
        self._trades_today: int = 0
        self._current_date: Optional[int] = None
        self._daily_realized: float = 0.0

    # ── Pre-trade gate ──────────────────────────────────────────────

    def check_pre_trade(
        self,
        bars: dict[str, DailyBar],
        trade: Trade,
        portfolio: "Portfolio",
    ) -> Optional[Trade]:
        """Validate one proposed trade. Returns (possibly shrunk) Trade or None.

        Rules are applied in order; the first rejecting rule wins. Shrinking
        rules mutate a copy of the trade and continue.
        """
        cfg = self.config
        bar = bars.get(trade.symbol)
        price = trade.price

        # Suspension: no trading on halted names.
        if cfg.reject_suspended and bar is not None and bar.is_suspended:
            return None

        # Price-limit lock: can't buy at limit-up (queue is all buyers),
        # can't sell at limit-down (queue is all sellers).
        if cfg.reject_limit_locked and bar is not None:
            if trade.quantity > 0 and bar.is_limit_up:
                return None
            if trade.quantity < 0 and bar.is_limit_down:
                return None

        # Circuit breaker: when halted, block new opens (closes still allowed).
        if self.halted and trade.quantity > 0 and trade.direction == "OPEN":
            return None

        # T+1: cap sells by available quantity.
        if cfg.enforce_t1 and trade.quantity < 0:
            pos = portfolio.get(trade.symbol)
            if pos.available_qty < abs(trade.quantity):
                sellable = int(pos.available_qty)
                if sellable <= 0:
                    return None
                trade = Trade(
                    date=trade.date, symbol=trade.symbol, price=price,
                    quantity=-sellable, direction=trade.direction,
                    cost=trade.cost, pnl=trade.pnl, reason=trade.reason,
                )

        # Board lot: buys must be multiples of 100.
        if cfg.enforce_lot and trade.quantity > 0:
            qty = round_lot_down(trade.quantity)
            if qty <= 0:
                return None
            if qty != trade.quantity:
                trade = Trade(
                    date=trade.date, symbol=trade.symbol, price=price,
                    quantity=qty, direction=trade.direction,
                    cost=trade.cost, pnl=trade.pnl, reason=trade.reason,
                )

        # Order frequency: max trades per day.
        if cfg.max_trades_per_day is not None:
            self._maybe_rollover_day(trade.date)
            if self._trades_today >= cfg.max_trades_per_day:
                return None

        # Notional cap: shrink a trade whose yuan value exceeds the limit.
        if cfg.max_notional_per_trade is not None:
            notional = price * abs(trade.quantity)
            if notional > cfg.max_notional_per_trade:
                max_qty = cfg.max_notional_per_trade / price
                if trade.quantity > 0:
                    max_qty = round_lot_down(max_qty) if cfg.enforce_lot else int(max_qty)
                    if max_qty <= 0:
                        return None
                    new_qty = max_qty
                else:
                    new_qty = -int(max_qty)
                trade = Trade(
                    date=trade.date, symbol=trade.symbol, price=price,
                    quantity=new_qty, direction=trade.direction,
                    cost=trade.cost, pnl=trade.pnl, reason=trade.reason,
                )

        # Single-name concentration: shrink a buy so the post-trade weight
        # in this name does not exceed ``max_weight_per_name``.
        if cfg.max_weight_per_name is not None and trade.quantity > 0:
            equity = portfolio.cash + sum(
                p.quantity * (bars[s].close if s in bars else p.avg_entry_price)
                for s, p in portfolio.positions.items() if p.quantity > 0
            )
            if equity <= 0:
                return None
            max_value = cfg.max_weight_per_name * equity
            pos = portfolio.get(trade.symbol)
            current_value = pos.quantity * price
            allowed_value = max_value - current_value
            if allowed_value <= 0:
                return None  # already at/over the cap
            allowed_qty = allowed_value / price
            if trade.quantity > allowed_qty:
                qty = round_lot_down(allowed_qty) if cfg.enforce_lot else int(allowed_qty)
                if qty <= 0:
                    return None
                trade = Trade(
                    date=trade.date, symbol=trade.symbol, price=price,
                    quantity=qty, direction=trade.direction,
                    cost=trade.cost, pnl=trade.pnl, reason=trade.reason,
                )

        return trade

    # ── Post-bar monitor ────────────────────────────────────────────

    def check_post_bar(
        self,
        date: int,
        bars: dict[str, DailyBar],
        portfolio: "Portfolio",
    ) -> list[Trade]:
        """Run all post-bar monitors; return forced-close trades.

        Checks (in order): per-position stop-loss → per-position take-profit
        → portfolio max-drawdown → daily loss limit. The first two may
        produce multiple forced closes (one per offending position); the
        latter two halt the system and close everything still held.
        """
        self._maybe_rollover_day(date)
        cfg = self.config
        forced: list[Trade] = []

        # Per-position stop-loss / take-profit.
        for sym, pos in list(portfolio.positions.items()):
            if pos.quantity <= 0:
                continue
            bar = bars.get(sym)
            price = bar.close if bar is not None else pos.avg_entry_price
            pos.mark_to_market(price)
            cost_basis = pos.quantity * pos.avg_entry_price
            if cost_basis <= 0:
                continue
            pnl_pct = pos.unrealized_pnl / cost_basis

            if cfg.stop_loss_pct is not None and pnl_pct < -cfg.stop_loss_pct:
                forced.append(self._force_close(date, sym, price, pos, "STOP_LOSS"))
                continue
            if cfg.take_profit_pct is not None and pnl_pct > cfg.take_profit_pct:
                forced.append(self._force_close(date, sym, price, pos, "TAKE_PROFIT"))

        # Apply any per-position forced closes to the portfolio before
        # evaluating portfolio-level drawdown / daily loss, so the equity
        # used for those checks reflects them.
        for t in forced:
            portfolio.apply_trade(t)
            self._daily_realized += t.pnl

        # Re-mark equity after the forced closes.
        equity = portfolio.cash + sum(
            p.quantity * (bars[s].close if s in bars else p.avg_entry_price)
            for s, p in portfolio.positions.items() if p.quantity > 0
        )
        if equity > self.peak_equity:
            self.peak_equity = equity

        capital = self.initial_capital or portfolio.initial_capital

        # Portfolio max drawdown: halt and liquidate.
        if cfg.max_drawdown_pct is not None and self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd > cfg.max_drawdown_pct:
                self.halted = True
                for sym, pos in list(portfolio.positions.items()):
                    if pos.quantity <= 0:
                        continue
                    bar = bars.get(sym)
                    price = bar.close if bar is not None else pos.avg_entry_price
                    forced.append(self._force_close(date, sym, price, pos, "MAX_DRAWDOWN"))
                return forced

        # Daily realized-loss limit: halt and liquidate.
        if cfg.daily_loss_limit_pct is not None and capital > 0:
            if self._daily_realized < -cfg.daily_loss_limit_pct * capital:
                self.halted = True
                for sym, pos in list(portfolio.positions.items()):
                    if pos.quantity <= 0:
                        continue
                    bar = bars.get(sym)
                    price = bar.close if bar is not None else pos.avg_entry_price
                    forced.append(self._force_close(date, sym, price, pos, "DAILY_LOSS_LIMIT"))
                return forced

        return forced

    # ── Helpers ─────────────────────────────────────────────────────

    def record_trade(self, date: int, symbol: str) -> None:
        """Record that a trade executed (for the per-day frequency cap)."""
        self._maybe_rollover_day(date)
        self._trades_today += 1

    def _maybe_rollover_day(self, date: int) -> None:
        """Reset per-day counters when the date advances."""
        if self._current_date != date:
            self._current_date = date
            self._trades_today = 0
            self._daily_realized = 0.0

    def _force_close(self, date: int, symbol: str, price: float, pos: "Position", reason: str) -> Trade:
        """Build a forced-close trade for an entire open position."""
        # T+1: only sell what's available today.
        qty = pos.available_qty
        direction = "CLOSE"
        return Trade(
            date=date, symbol=symbol, price=price,
            quantity=-qty, direction=direction,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset all risk state (peak equity, day counters, halt flag)."""
        self.peak_equity = 0.0
        self.halted = False
        self._trades_today = 0
        self._current_date = None
        self._daily_realized = 0.0
