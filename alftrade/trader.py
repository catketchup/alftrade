"""
trader.py - Two-phase daily trading loop (next-open execution).

Ties the ``TradingEngine`` (signals), ``PortfolioManager`` (target orders), and
``Broker`` (execution) into a realistic next-open flow:

    Day t open:  execute yesterday's orders (slippage / volume caps) → settle
    Day t close: compute signals → plan target orders (deferred to next open)

This removes the same-bar look-ahead bias of ``PortfolioManager.decide``.
"""

from typing import Optional

from alftrade.engine import TradingEngine
from alftrade.portfolio import PortfolioManager, Trade
from alftrade.broker import Broker, ExecutionConfig, OrderRecord


class Trader:
    """Orchestrates the daily decide → (next-open) execute cycle.

    Holds the orders placed at the previous close (``pending``) and executes
    them at the next day's open through the ``Broker``.

    Args:
        engine: ``TradingEngine`` producing per-stock signals.
        portfolio: ``PortfolioManager`` producing target orders / settling fills.
        broker: ``Broker`` (or None → default) simulating fills.
    """

    def __init__(
        self,
        engine: TradingEngine,
        portfolio: PortfolioManager,
        broker: Optional[Broker] = None,
    ):
        self.engine = engine
        self.portfolio = portfolio
        self.broker = broker or Broker()
        self.pending: list[Trade] = []
        # Journal of every order attempt (FILLED / PARTIAL / REJECTED / SKIPPED).
        self.order_log: list[OrderRecord] = []

    def step(self, date: int, bars: dict) -> list[Trade]:
        """Run one trading day: execute pending orders, then plan new ones.

        Args:
            date: Trading-day label.
            bars: ``{symbol: DailyBar}`` for this date.

        Returns:
            The fills executed at this day's open.
        """
        # Phase 1: execute yesterday's orders at open[date].
        # When the account is halted (drawdown / daily-loss limit triggered
        # at the prior close), only liquidations execute: pending buys are
        # cancelled, exactly like cancelling open orders after a halt.
        pending = self.pending
        if self.portfolio.risk.halted:
            pending = [t for t in pending if t.quantity < 0]
        result = self.broker.execute(date, pending, bars)
        self.order_log.extend(result.records)
        fills = self.portfolio.settle(date, result.fills)
        self.pending = []

        # Phase 2: plan new orders at close[date] (execute at open[date+1]).
        signals = self.engine.on_cross_section(date, bars)
        self.pending = self.portfolio.plan(date, bars, signals)
        return fills

    @property
    def trades(self) -> list[Trade]:
        """All filled trades so far (delegates to the PortfolioManager)."""
        return self.portfolio.trades

    def set_execution_timing(self, timing: str) -> str:
        """Manually switch the execution timing; returns the previous value.

        Timing is one of "open", "close_auction", "vwap".  The switch takes
        effect on the *next* fill: orders already pending execute under the
        new timing (the Broker reads the config at fill time).
        """
        if timing not in ("open", "close_auction", "vwap"):
            raise ValueError(f"Unknown execution_timing: {timing!r}")
        prev = self.broker.config.execution_timing
        self.broker.config.execution_timing = timing
        return prev

    def to_state(self) -> dict:
        """JSON-serializable snapshot: pending orders + order journal.

        The engine and portfolio are rebuilt by the caller (the engine needs
        the market panel and the stored calibration; the portfolio manager is
        rebuilt via ``PortfolioManager.from_state``), so only the trader's own
        state is serialized here.
        """
        from alftrade.broker import record_to_dict

        return {
            "pending": [t.to_dict() for t in self.pending],
            "order_log": [record_to_dict(r) for r in self.order_log],
        }

    @classmethod
    def from_state(cls, state: dict, engine: TradingEngine,
                   portfolio: PortfolioManager) -> "Trader":
        from alftrade.broker import record_from_dict

        trader = cls(engine, portfolio)
        trader.pending = [Trade.from_dict(t) for t in state.get("pending", [])]
        trader.order_log = [record_from_dict(r) for r in state.get("order_log", [])]
        return trader

    def reset(self) -> None:
        """Clear pending orders and reset the portfolio + risk state."""
        self.pending = []
        self.order_log = []
        self.portfolio.reset()
