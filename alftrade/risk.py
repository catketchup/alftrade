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
        - Board lot: reject/shrink buys that aren't a multiple of 100.
        - Single-name concentration: shrink buys that would breach
          ``max_weight_per_name``.
        - Notional cap: shrink any trade exceeding ``max_notional_per_trade``.
        - Order frequency: reject once ``max_trades_per_day`` is hit.
        - Circuit breaker: reject new opens when the portfolio is halted.

     (T+1 settlement is *not* a pre-trade rule — it is enforced at execution
     time by ``Position.apply_trade``, after the daily rollover, so that
     next-open execution is not wrongly delayed by a stale ``available_qty``.)

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
    # (T+1 is not a pre-trade rule — it is enforced at execution time by
    #  Position.apply_trade, after the daily rollover.)
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

        # Circuit breaker: when halted, block ALL buys (new opens AND adds to
        # existing positions — a halted account only liquidates). Closes pass.
        if self.halted and trade.quantity > 0:
            return None

        # NOTE: T+1 is NOT enforced here.  It is an execution-time constraint
        # (can I sell these shares *today*?) and is applied by
        # ``Position.apply_trade`` when the fill settles, after the daily
        # rollover.  Enforcing it at plan time would use a stale
        # ``available_qty`` (the shares bought at today's open are still locked
        # at the close, but will be unlocked by the next open's rollover) and
        # wrongly delay next-open sells by a day.

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
        """Run all post-bar monitors; return force-close *orders*.

        Checks (in order): per-position stop-loss → per-position take-profit
        → portfolio max-drawdown → daily loss limit.  This is **decide-only**:
        the returned trades are orders to execute at the next open, not fills.
        The caller applies them (via ``PortfolioManager.settle``) and reports
        the realized PnL back through ``accumulate_realized``.
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

        # Mark-to-market equity (do NOT apply the forced closes — they execute
        # at the next open).
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

    def accumulate_realized(self, pnl: float) -> None:
        """Record realized PnL from a fill for the daily-loss monitor."""
        self._daily_realized += pnl

    def to_state(self) -> dict:
        """JSON-serializable snapshot of the running risk state.

        The ``config`` is not serialized (it is a constructor argument) —
        ``from_state`` takes it explicitly.
        """
        return {
            "peak_equity": self.peak_equity,
            "halted": self.halted,
            "trades_today": self._trades_today,
            "current_date": self._current_date,
            "daily_realized": self._daily_realized,
            "initial_capital": self.initial_capital,
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        config: Optional[RiskConfig] = None,
        initial_capital: Optional[float] = None,
    ) -> "RiskManager":
        rm = cls(config=config, initial_capital=initial_capital)
        rm.peak_equity = state.get("peak_equity", 0.0)
        rm.halted = state.get("halted", False)
        rm._trades_today = state.get("trades_today", 0)
        rm._current_date = state.get("current_date")
        rm._daily_realized = state.get("daily_realized", 0.0)
        if state.get("initial_capital") is not None and initial_capital is None:
            rm.initial_capital = state["initial_capital"]
        return rm

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
        """Build a forced-close trade for an entire open position.

        Uses the full ``quantity`` (not ``available_qty``): T+1 is an
        execution-time constraint enforced by ``Position.apply_trade`` after
        the daily rollover, so capping here would emit a zero/partial order
        that never actually closes the position.
        """
        direction = "CLOSE"
        return Trade(
            date=date, symbol=symbol, price=price,
            quantity=-pos.quantity, direction=direction,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset all risk state (peak equity, day counters, halt flag)."""
        self.peak_equity = 0.0
        self.halted = False
        self._trades_today = 0
        self._current_date = None
        self._daily_realized = 0.0
