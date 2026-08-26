"""
broker.py - Execution simulation layer (next-open, slippage, volume caps).

Simulates the fill side of trading without a live API.  The ``PortfolioManager``
produces *target orders* at day t's close, and the ``Broker`` fills them at day
t+1's open with realistic frictions:

  * next-open execution (no same-bar look-ahead),
  * slippage = fixed bps + half-spread + market impact ∝ order share of volume,
  * fills happen in the open *call auction*: the participation cap applies to
    auction liquidity (a few % of the day's volume), not the whole day,
  * price-limit lock checked against the OPEN price (not the close),
  * suspension / missing-bar skip,
  * transaction costs (commission + stamp duty) via ``CostModel``,
  * a full order journal: every order ends FILLED / PARTIAL / REJECTED /
    SKIPPED with a reason, so paper trading can report unfilled orders.

Orders are one-shot: an unfilled remainder is not carried as a stale order —
the next day's re-plan naturally issues a fresh order for the missing shares.
"""

from dataclasses import dataclass
from typing import Optional

from alftrade.portfolio import Trade, round_lot_down


@dataclass
class ExecutionConfig:
    """Configuration for the execution simulator.

    Attributes:
        slippage_bps: Fixed slippage applied to the execution price (decimal).
        half_spread_bps: Half the bid-ask spread, also applied (decimal).
        max_participation: Max fraction of *open-auction liquidity* a single
            order may take. Larger orders fill partially and the remainder is
            re-ordered next day via re-planning.
        reject_limit_locked: Reject buys whose open is at limit-up and sells
            whose open is at limit-down (queue positions cannot fill).
        auction_volume_frac: Open-auction volume ≈ this fraction of the day's
            total volume (A-share call auction is typically 3-10 %). The
            participation cap applies to auction liquidity only.
        impact_k: Market-impact coefficient — extra bps of slippage per 1 %
            of the day's volume the order represents (larger orders pay more).
        execution_timing: When fills happen, given orders placed at the prior
            close:
              * "open"           — next-day open call auction (price = open).
              * "close_auction"  — next-day closing auction (price = close),
                carrying the full day's move.
              * "vwap"           — algorithm spread through the day (price =
                day VWAP = amount / volume), carrying about half the day's move.
            Switching the attribute on an existing broker takes effect on the
            next fill (pending orders execute under the new timing).
    """
    slippage_bps: float = 0.0025
    half_spread_bps: float = 0.0005
    max_participation: float = 0.20
    reject_limit_locked: bool = True
    auction_volume_frac: float = 0.10
    impact_k: float = 2.0
    execution_timing: str = "open"


@dataclass
class OrderRecord:
    """Outcome of one order attempt, for the order journal.

    Attributes:
        date: Trading day of the attempt.
        symbol: Ticker.
        side: "BUY" or "SELL".
        requested_qty: Signed quantity requested.
        filled_qty: Signed quantity actually filled (0 when not filled).
        price: Execution price, or 0.0 when not filled.
        status: "FILLED", "PARTIAL", "REJECTED" or "SKIPPED".
        reason: Human-readable explanation (limit-lock, volume cap, ...).
    """
    date: int
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float = 0.0
    price: float = 0.0
    status: str = "SKIPPED"
    reason: str = ""


@dataclass
class ExecutionResult:
    """Returned by ``Broker.execute``: fills plus the full order journal."""

    fills: list[Trade]
    records: list[OrderRecord]


def record_to_dict(r: OrderRecord) -> dict:
    """JSON-serializable snapshot of an OrderRecord (for the order journal)."""
    return {
        "date": r.date, "symbol": r.symbol, "side": r.side,
        "requested_qty": r.requested_qty, "filled_qty": r.filled_qty,
        "price": r.price, "status": r.status, "reason": r.reason,
    }


def record_from_dict(d: dict) -> OrderRecord:
    return OrderRecord(
        date=d["date"], symbol=d["symbol"], side=d["side"],
        requested_qty=d["requested_qty"], filled_qty=d.get("filled_qty", 0.0),
        price=d.get("price", 0.0), status=d.get("status", "SKIPPED"),
        reason=d.get("reason", ""),
    )


class Broker:
    """Fills target orders at next-open prices with execution frictions.

    ``execute`` is stateless: it takes a batch of orders (placed at the previous
    close) and the current day's bars, and returns the fills that actually
    occurred plus one ``OrderRecord`` per order attempt.  Orders are
    represented as ``Trade`` objects with a *reference* price (the previous
    close, used for target sizing); fills are new ``Trade`` objects carrying
    the actual execution price, quantity, and cost.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None, cost_model=None):
        from alftrade.portfolio import CostModel

        self.config = config or ExecutionConfig()
        self.cost_model = cost_model or CostModel()

    def execute(
        self,
        date: int,
        orders: list[Trade],
        bars: dict,
    ) -> ExecutionResult:
        """Execute a batch of orders at day ``date``'s open.

        Args:
            date: Trading day of execution (orders were placed the prior close).
            orders: Target orders (``Trade`` objects, signed quantity).
            bars: ``{symbol: DailyBar}`` for the execution day.

        Returns:
            ``ExecutionResult(fills, records)`` — the fills that actually
            occurred (may be fewer / smaller than the orders) plus the full
            order journal.
        """
        fills: list[Trade] = []
        records: list[OrderRecord] = []
        for order in orders:
            fill, record = self._fill_one(date, order, bars)
            records.append(record)
            if fill is not None:
                fills.append(fill)
        return ExecutionResult(fills=fills, records=records)

    def _fill_one(
        self,
        date: int,
        order: Trade,
        bars: dict,
    ) -> tuple[Optional[Trade], OrderRecord]:
        """Attempt to fill one order; return ``(fill_or_None, record)``."""
        cfg = self.config
        bar = bars.get(order.symbol)
        side = "BUY" if order.quantity > 0 else "SELL"

        if bar is None:
            return None, OrderRecord(date, order.symbol, side, order.quantity,
                                     reason="no bar for symbol")
        if bar.is_suspended or bar.volume <= 0:
            return None, OrderRecord(date, order.symbol, side, order.quantity,
                                     reason="suspended / zero volume")

        # Execution timing determines the reference price, the price used for
        # the limit-lock check, and the liquidity pool for participation caps.
        timing = cfg.execution_timing
        if timing == "open":
            ref_price, lock_price = bar.open, bar.open
            liquid = bar.volume * cfg.auction_volume_frac
        elif timing == "close_auction":
            ref_price, lock_price = bar.close, bar.close
            liquid = bar.volume * cfg.auction_volume_frac
        elif timing == "vwap":
            ref_price = bar.amount / bar.volume if bar.amount > 0 else bar.close
            lock_price = bar.open  # open not locked → intraday opportunity
            liquid = bar.volume    # spread through the whole session
        else:
            raise ValueError(f"Unknown execution_timing: {timing!r}")

        # Price-limit lock checked against the price that governs the timing:
        # at limit-up only sellers are matched, at limit-down only buyers.
        if cfg.reject_limit_locked:
            if side == "BUY" and bar.limit_up_price > 0 and lock_price >= bar.limit_up_price - 1e-9:
                return None, OrderRecord(date, order.symbol, side, order.quantity,
                                         status="REJECTED",
                                         reason=f"limit-up at {timing}")
            if side == "SELL" and bar.limit_down_price > 0 and lock_price <= bar.limit_down_price + 1e-9:
                return None, OrderRecord(date, order.symbol, side, order.quantity,
                                         status="REJECTED",
                                         reason=f"limit-down at {timing}")

        # Market impact proportional to the order's share of the *liquidity
        # pool* for this timing (an auction order hits the thin auction book
        # much harder than the same order spread over the whole day).
        impact_bps = cfg.impact_k * (abs(order.quantity) / liquid) * 100.0
        impact = cfg.slippage_bps + cfg.half_spread_bps + impact_bps / 10_000.0
        price = ref_price * (1.0 + impact) if side == "BUY" else ref_price * (1.0 - impact)
        price = round(price, 2)

        # Volume-participation cap on this timing's liquidity pool.
        max_qty = cfg.max_participation * liquid
        qty = min(abs(order.quantity), max_qty)
        if side == "BUY":
            qty = round_lot_down(qty)   # buys must be whole board lots
        else:
            qty = int(qty)              # sells may be odd lots (to fully close)
        if qty <= 0:
            return None, OrderRecord(date, order.symbol, side, order.quantity,
                                     status="REJECTED",
                                     reason="volume cap (liquidity exhausted)")

        signed_qty = qty if side == "BUY" else -qty
        cost = self.cost_model.cost(side, price, qty)
        fill = Trade(
            date=date, symbol=order.symbol, price=price,
            quantity=signed_qty, direction=order.direction,
            cost=cost, pnl=0.0, reason=order.reason,
        )
        if abs(qty) >= abs(order.quantity) - 1e-9:
            status, reason = "FILLED", "filled"
        else:
            status, reason = "PARTIAL", "partial fill (volume cap)"
        record = OrderRecord(date, order.symbol, side, order.quantity,
                             signed_qty, price, status, reason)
        return fill, record
