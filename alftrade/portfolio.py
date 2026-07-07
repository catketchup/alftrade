"""
portfolio.py - Multi-stock portfolio, signal aggregation, and the
cross-sectional rebalancing manager.

This is the daily, long-only, multi-stock analogue of hftrade's
``position.py`` (which was single-stock and allowed shorting). The pipeline:

    TradingEngine.on_cross_section(date, bars)
        → {symbol: [Signal]}
        → SignalAggregator.aggregate(symbol, signals) → CompositeSignal[s]
        → PortfolioManager.decide(date, bars, composites)
        → target weights  →  proposed Trades
        → RiskManager.check_pre_trade()   (T+1 / price-limit / lot / cap)
        → Portfolio.apply_trade()
        → RiskManager.check_post_bar()    (stop-loss / drawdown / daily-loss)

A-share specifics modelled here
-------------------------------
  * **Long-only**: ``Position.quantity`` is always ≥ 0. The ``apply_trade``
    path for sells refuses to oversell (which would create a short).
  * **T+1 settlement**: shares bought on day t cannot be sold until t+1.
    Each ``Position`` tracks ``available_qty = quantity - t1_locked``. The
    locked amount clears at the start of the next trading day via
    ``Portfolio.rollover_t1()``. The risk layer enforces that sell quantity
    never exceeds ``available_qty``.
  * **Lot size**: buys must be a multiple of the board lot (100 shares).
    Sells may include an odd lot only to fully flatten a position.
  * **Transaction costs**: a ``CostModel`` (commission both ways + stamp
    duty on sells) is applied per trade and deducted from realized PnL.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from alftrade.data import DailyBar
from alftrade.engine import Signal

if TYPE_CHECKING:
    from alftrade.calibration import CalibrationResult
    from alftrade.risk import RiskManager


# A-share board lot: orders to *buy* must be in multiples of this many shares.
BOARD_LOT = 100


def round_lot_down(qty: float, lot: int = BOARD_LOT) -> int:
    """Round a buy quantity down to the nearest board lot (≥ 0)."""
    return max(0, int(qty // lot) * lot)


# ── Cost model ──────────────────────────────────────────────────────────


@dataclass
class CostModel:
    """A-share transaction cost model.

    Defaults reflect typical retail institutional rates:
        Commission: 2.5 bps on both buys and sells, with a 5-yuan minimum.
        Stamp duty: 10 bps on *sells only* (the seller side).
        Transfer fee: 0.2 bps both ways (Shanghai only; bundled here).

    All rates are in decimal (e.g. 0.00025 = 2.5 bps).
    """
    commission_bps: float = 0.00025
    stamp_duty_bps: float = 0.001
    transfer_fee_bps: float = 0.00002
    min_commission: float = 5.0

    def cost(self, side: str, price: float, quantity: float) -> float:
        """Compute the total cost (yuan) for one trade.

        Args:
            side: "BUY" or "SELL".
            price: Execution price.
            quantity: Absolute share count.
        """
        if quantity <= 0:
            return 0.0
        notional = price * quantity
        commission = max(notional * self.commission_bps, self.min_commission)
        transfer = notional * self.transfer_fee_bps
        stamp = notional * self.stamp_duty_bps if side == "SELL" else 0.0
        return commission + transfer + stamp


# ── Trade & Position ────────────────────────────────────────────────────


@dataclass
class Trade:
    """Record of a single execution for one stock.

    Attributes:
        date: Trading day the trade was booked.
        symbol: Ticker.
        price: Execution price (the bar's close, per the engine convention).
        quantity: Signed share count. Positive = buy, negative = sell.
                  Sells are constrained to never exceed available_qty.
        direction: "OPEN" / "INCREASE" / "REDUCE" / "CLOSE".
        cost: Transaction cost paid (yuan). Deducted from realized PnL.
        pnl: Realized PnL on the closed portion (0 for opens/increases).
        reason: Why the trade happened: "SIGNAL", "STOP_LOSS",
                "TAKE_PROFIT", "DRAWDOWN", "DAILY_LOSS_LIMIT", "REBALANCE".
    """
    date: int
    symbol: str
    price: float
    quantity: float
    direction: str
    cost: float = 0.0
    pnl: float = 0.0
    reason: str = "SIGNAL"

    def __repr__(self):
        return (f"Trade({self.date} {self.symbol} {self.direction} "
                f"qty={self.quantity:+.0f} @ {self.price:.2f} "
                f"cost={self.cost:.2f} pnl={self.pnl:.2f} [{self.reason}])")


@dataclass
class Position:
    """Long-only holdings for a single stock, with T+1 lock tracking.

    Attributes:
        quantity: Total shares held (≥ 0).
        avg_entry_price: Weighted average buy price.
        t1_locked: Shares bought *today* and not yet settled. Cannot be sold
                   until ``rollover_t1`` clears the lock at the next bar.
        realized_pnl: Cumulative realized profit (net of costs).
        unrealized_pnl: Mark-to-market on the open position at the last bar.
    """
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    t1_locked: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity <= 1e-9

    @property
    def available_qty(self) -> float:
        """Shares that can be sold today under T+1 (= quantity − locked)."""
        return max(0.0, self.quantity - self.t1_locked)

    def mark_to_market(self, price: float) -> None:
        """Revalue the open position at the current bar's price."""
        if self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0

    def rollover_t1(self) -> None:
        """Clear the T+1 lock: yesterday's buys become sellable today."""
        self.t1_locked = 0.0

    def apply_trade(self, trade: Trade) -> None:
        """Update position state after a trade fill (long-only).

        * Buys: increase quantity, update weighted avg entry, lock the bought
          shares under T+1.
        * Sells: realize PnL on the closed portion, deduct cost, and refuse
          to oversell (which would create a short).
        """
        if trade.quantity == 0.0:
            return

        if trade.quantity > 0:
            # Buy: weighted average entry; lock bought shares for T+1.
            total_cost = self.avg_entry_price * self.quantity + trade.price * trade.quantity
            self.quantity += trade.quantity
            self.avg_entry_price = total_cost / self.quantity if self.quantity > 0 else 0.0
            self.t1_locked += trade.quantity
            self.realized_pnl -= trade.cost
            if trade.direction == "OPEN":
                pass  # opening, no PnL
            return

        # Sell: long-only, cannot sell more than available_qty (T+1).
        sell_qty = min(-trade.quantity, self.available_qty)
        if sell_qty <= 0:
            return  # nothing sellable today (T+1 blocked)

        pnl_per_share = trade.price - self.avg_entry_price
        self.realized_pnl += pnl_per_share * sell_qty - trade.cost
        self.quantity -= sell_qty
        # Sells draw from the non-locked portion first; t1_locked only
        # decreases if the sell exceeded the unlocked shares (shouldn't
        # happen in once-per-day rebalance, but kept for correctness).
        unlocked = self.quantity + sell_qty - self.t1_locked
        if sell_qty > unlocked:
            self.t1_locked = max(0.0, self.t1_locked - (sell_qty - unlocked))
        if self.quantity <= 1e-9:
            self.quantity = 0.0
            self.avg_entry_price = 0.0
            self.t1_locked = 0.0


# ── Portfolio (collection of positions) ─────────────────────────────────


@dataclass
class Portfolio:
    """Container of per-symbol positions plus cash and equity tracking.

    Attributes:
        cash: Free cash (yuan). Buys decrease it, sells increase it.
        initial_capital: Starting cash, for drawdown / return reporting.
        positions: {symbol: Position}.
    """
    cash: float
    initial_capital: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if self.initial_capital == 0.0:
            self.initial_capital = self.cash

    def get(self, symbol: str) -> Position:
        """Get or create the position for `symbol`."""
        if symbol not in self.positions:
            self.positions[symbol] = Position()
        return self.positions[symbol]

    def apply_trade(self, trade: Trade) -> None:
        """Apply a trade to the position and to cash."""
        pos = self.get(trade.symbol)
        # Cash impact: buy → pay notional + cost; sell → receive notional − cost.
        # cost is already applied to realized_pnl inside Position.apply_trade,
        # so here we only move the notional and *also* subtract the cost from
        # cash (cost is a real cash outflow). To avoid double-counting cost
        # in PnL reporting, realized_pnl nets cost and cash nets cost too;
        # total equity = cash + market_value remains consistent because the
        # cost leaves both cash and (eventually) realized_pnl.
        notional = trade.price * abs(trade.quantity)
        if trade.quantity > 0:
            self.cash -= notional + trade.cost
        else:
            self.cash += notional - trade.cost
        pos.apply_trade(trade)

    def mark_to_market(self, bars: dict[str, DailyBar]) -> float:
        """Revalue every open position at its bar's close; return total equity.

        Also rolls T+1 locks forward: callers should invoke ``rollover_t1``
        once per new bar *before* generating trades (see PortfolioManager).
        """
        equity = self.cash
        for sym, pos in self.positions.items():
            bar = bars.get(sym)
            price = bar.close if bar is not None else pos.avg_entry_price
            pos.mark_to_market(price)
            equity += pos.quantity * price
        return equity

    def rollover_t1(self) -> None:
        """Clear T+1 locks on every position (call at the start of a new day)."""
        for pos in self.positions.values():
            pos.rollover_t1()

    @property
    def total_pnl(self) -> float:
        """Realized + unrealized PnL across all positions (net of costs)."""
        return sum(p.realized_pnl + p.unrealized_pnl for p in self.positions.values())

    @property
    def market_value(self) -> float:
        """Current market value of all open positions."""
        return sum(p.quantity * p.avg_entry_price for p in self.positions.values()
                   if p.quantity > 0)  # approx; use mark_to_market for accurate


# ── Signal aggregation ──────────────────────────────────────────────────


@dataclass
class CompositeSignal:
    """Aggregated view of all signals for one stock on one date.

    Attributes:
        symbol: Ticker.
        score: Combined predicted-return score. Positive → long candidate.
        contributing: Number of individual signals aggregated.
        direction: +1 if score > 0 else 0 (long-only convention).
    """
    symbol: str
    score: float
    contributing: int
    direction: int = 0


class SignalAggregator:
    """Combines a stock's per-factor signals into one composite score.

    Weight resolution per signal name (same priority as hftrade):
      1. Manual ``weights`` dict entry → use it.
      2. ``calibration`` provided → use |IC| (data-driven quality weight).
      3. Otherwise → 1.0.

    In calibrated mode the engine emits a single "model" signal whose
    strength is already the predicted return, so aggregation is trivial
    (one contributor) and the composite score *is* the model's prediction.

    Args:
        weights: Per-signal-name manual weight overrides.
        calibration: Optional CalibrationResult for IC-based weights.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        calibration: "Optional[CalibrationResult]" = None,
    ):
        self.weights = weights or {}
        self.calibration = calibration

    def _resolve_weight(self, signal_name: str) -> float:
        """Weight by priority: manual > calibration IC (if known) > 1.0.

        Note: a signal whose name is *not* in ``calibration.ics`` falls back
        to 1.0 rather than 0.0. This matters in calibrated mode, where the
        engine emits a single signal named "model" (the whole-factor-model
        prediction) — it is not a per-factor IC key, so it must default to
        weight 1.0 rather than being zeroed out.
        """
        if signal_name in self.weights:
            return self.weights[signal_name]
        if self.calibration is not None and signal_name in self.calibration.ics:
            return abs(self.calibration.ics[signal_name])
        return 1.0

    def aggregate(self, symbol: str, signals: list[Signal]) -> CompositeSignal:
        """Combine one stock's signals into a composite score.

        Score = Σ (direction × strength × weight). Long-only direction is
        determined by the sign of the score (0 if score ≤ 0).
        """
        if not signals:
            return CompositeSignal(symbol, 0.0, 0, 0)
        score = 0.0
        count = 0
        for s in signals:
            w = self._resolve_weight(s.name)
            score += s.direction * s.strength * w
            count += 1
        direction = 1 if score > 0 else 0
        return CompositeSignal(symbol, score, count, direction)

    def aggregate_all(self, signals_by_symbol: dict[str, list[Signal]]) -> dict[str, CompositeSignal]:
        """Aggregate every stock's signals; return {symbol: CompositeSignal}.

        Stocks with no signals are not added to the result (treated as FLAT).
        """
        return {sym: self.aggregate(sym, sigs) for sym, sigs in signals_by_symbol.items()}


# ── Portfolio manager (cross-sectional rebalancing) ─────────────────────


class PortfolioManager:
    """Converts composite scores into a target portfolio and rebalance trades.

    Strategy (cross-sectional long-only):
      1. Rank stocks by composite score.
      2. Pick the top ``n_holdings`` with positive score.
      3. Assign target weights: equal-weight by default, or score-weighted
         if ``weighting="score"``.
      4. For each stock, compute the trade needed to move from its current
         weight to the target weight, subject to:
           - available cash (buys capped by cash),
           - T+1 availability (sells capped by available_qty),
           - board-lot rounding (buys rounded to 100s),
           - the RiskManager's pre-trade gate.

    Args:
        capital: Total equity used for weight→share conversion. Defaults to
            the portfolio's current equity if None.
        n_holdings: Number of long positions to hold.
        weighting: "equal" or "score".
        aggregator: SignalAggregator to combine per-factor signals.
        risk_manager: Optional RiskManager; a default one is created if None.
        cost_model: Transaction cost model.
        top_quantile: If set (0,1), pick top quantile of universe instead of
            a fixed n_holdings. E.g. 0.2 = top 20%.
    """

    def __init__(
        self,
        capital: Optional[float] = None,
        n_holdings: int = 10,
        weighting: str = "equal",
        aggregator: Optional[SignalAggregator] = None,
        risk_manager: "Optional[RiskManager]" = None,
        cost_model: Optional[CostModel] = None,
        top_quantile: Optional[float] = None,
    ):
        self.capital = capital
        self.n_holdings = n_holdings
        self.weighting = weighting
        self.aggregator = aggregator or SignalAggregator()
        self.cost_model = cost_model or CostModel()
        self.top_quantile = top_quantile
        # Import lazily to avoid a circular import with risk.py.
        if risk_manager is None:
            from alftrade.risk import RiskManager
            risk_manager = RiskManager()
        self.risk = risk_manager
        self.portfolio = Portfolio(cash=capital or 0.0)
        self.trades: list[Trade] = []
        # Track the last decision date so T+1 locks only roll over when the
        # date actually advances (calling decide() twice on the same date
        # must NOT unlock same-day purchases).
        self._last_date: Optional[int] = None

    def decide(
        self,
        date: int,
        bars: dict[str, DailyBar],
        signals_by_symbol: dict[str, list[Signal]],
    ) -> list[Trade]:
        """Run one day's rebalance: aggregate → target weights → trades.

        Pipeline:
          1. Rollover T+1 locks (yesterday's buys are now sellable).
          2. Mark every position to market at today's close.
          3. Aggregate per-stock signals → composite scores.
          4. Build target weights from top-scored stocks.
          5. Generate proposed buy/sell trades to reach targets.
          6. Pass each through RiskManager.check_pre_trade (may reject/shrink).
          7. Execute approved trades on the portfolio.
          8. Run RiskManager.check_post_bar (stop-loss / drawdown / etc.).
        """
        # 1. New day: clear T+1 locks — but only when the date advances.
        #    Calling decide() twice on the same date must NOT unlock same-day
        #    purchases (T+1 is a calendar rule, not a call-frequency rule).
        if self._last_date != date:
            self.portfolio.rollover_t1()
            self._last_date = date

        # 2. Mark to market.
        equity = self.portfolio.mark_to_market(bars)
        capital = self.capital or equity

        # 3. Aggregate.
        composites = self.aggregator.aggregate_all(signals_by_symbol)

        # 4. Target weights.
        targets = self._target_weights(composites, bars, capital)

        # 5. Proposed trades: sells first (free up cash), then buys.
        proposed = self._propose_trades(date, bars, targets, capital)
        executed: list[Trade] = []

        # 6-7. Pre-trade risk gate + execution.
        for trade in proposed:
            approved = self.risk.check_pre_trade(bars, trade, self.portfolio)
            if approved is not None:
                # Re-price cost after any qty shrink the risk manager did.
                side = "BUY" if approved.quantity > 0 else "SELL"
                approved.cost = self.cost_model.cost(side, approved.price, abs(approved.quantity))
                self.portfolio.apply_trade(approved)
                self.risk.record_trade(date, approved.symbol)
                self.trades.append(approved)
                executed.append(approved)

        # Re-mark after signal-driven trades.
        self.portfolio.mark_to_market(bars)

        # 8. Post-bar risk monitor (may force-close positions).
        forced = self.risk.check_post_bar(date, bars, self.portfolio)
        for ft in forced:
            side = "SELL"
            ft.cost = self.cost_model.cost(side, ft.price, abs(ft.quantity))
            self.portfolio.apply_trade(ft)
            self.risk.record_trade(date, ft.symbol)
            self.trades.append(ft)
            executed.append(ft)

        self.portfolio.mark_to_market(bars)
        return executed

    def _target_weights(
        self,
        composites: dict[str, CompositeSignal],
        bars: dict[str, DailyBar],
        capital: float,
    ) -> dict[str, float]:
        """Pick the long basket and assign target weights (summing to ≤ 1)."""
        # Only long candidates: positive score, not suspended, not at limit-up.
        candidates = [
            c for c in composites.values()
            if c.direction > 0 and c.symbol in bars and not bars[c.symbol].is_suspended
        ]
        # Filter out limit-up names (can't buy).
        candidates = [c for c in candidates if not bars[c.symbol].is_limit_up]
        candidates.sort(key=lambda c: c.score, reverse=True)

        n_pick = self._n_to_pick(len(bars))
        picked = candidates[:n_pick]
        if not picked:
            return {}

        if self.weighting == "score":
            total = sum(c.score for c in picked)
            if total <= 1e-12:
                weights = {c.symbol: 1.0 / len(picked) for c in picked}
            else:
                weights = {c.symbol: c.score / total for c in picked}
        else:  # equal
            weights = {c.symbol: 1.0 / len(picked) for c in picked}
        return weights

    def _n_to_pick(self, universe_size: int) -> int:
        if self.top_quantile is not None:
            return max(1, int(universe_size * self.top_quantile))
        return self.n_holdings

    def _propose_trades(
        self,
        date: int,
        bars: dict[str, DailyBar],
        targets: dict[str, float],
        capital: float,
    ) -> list[Trade]:
        """Generate trades to move from current to target weights.

        Sells first (positions not in targets, or overweight), then buys for
        targets (underweight or new). Buys are capped by available cash and
        rounded to the board lot.
        """
        proposed: list[Trade] = []
        prices = {s: b.close for s, b in bars.items()}

        # Current weights by market value.
        mvs = {s: p.quantity * prices.get(s, p.avg_entry_price)
               for s, p in self.portfolio.positions.items() if p.quantity > 0}
        total_mv = sum(mvs.values()) + self.portfolio.cash
        if total_mv <= 0:
            return []

        # ── Sells: drop names not in targets, or trim overweight ones ──
        for sym, pos in list(self.portfolio.positions.items()):
            if pos.quantity <= 0:
                continue
            price = prices.get(sym, pos.avg_entry_price)
            target_value = targets.get(sym, 0.0) * total_mv
            current_value = pos.quantity * price
            if current_value > target_value + 1e-6:
                excess_value = current_value - target_value
                # Cap sell by T+1 availability.
                sell_shares = min(excess_value / price, pos.available_qty)
                # If fully dropping the name, allow selling down to 0 (odd lot ok).
                if target_value <= 1e-6:
                    sell_shares = min(pos.quantity, pos.available_qty)
                sell_shares = int(sell_shares)  # whole shares only
                if sell_shares > 0:
                    direction = "CLOSE" if target_value <= 1e-6 else "REDUCE"
                    proposed.append(Trade(
                        date=date, symbol=sym, price=price,
                        quantity=-sell_shares, direction=direction,
                        reason="REBALANCE",
                    ))

        # ── Buys: raise underweight / open new target names ──
        # Estimate cash after the proposed sells (rough; risk gate will
        # re-check). This is an upper bound on buy power.
        pending_sell_proceeds = sum(
            t.price * abs(t.quantity) for t in proposed if t.quantity < 0
        )
        buyable_cash = self.portfolio.cash + pending_sell_proceeds

        # Sort buys by target weight descending so large allocations get
        # priority when cash is tight.
        buy_list = sorted(targets.items(), key=lambda kv: kv[1], reverse=True)
        for sym, w in buy_list:
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            pos = self.portfolio.get(sym)
            target_value = w * total_mv
            current_value = pos.quantity * price
            needed_value = target_value - current_value
            if needed_value <= 1e-6:
                continue
            # Cap by available cash and round to board lot.
            raw_qty = min(needed_value / price, buyable_cash / price) if buyable_cash > 0 else 0
            qty = round_lot_down(raw_qty)
            if qty <= 0:
                continue
            direction = "OPEN" if pos.is_flat else "INCREASE"
            proposed.append(Trade(
                date=date, symbol=sym, price=price,
                quantity=qty, direction=direction,
                reason="REBALANCE",
            ))
            buyable_cash -= qty * price

        return proposed

    @property
    def total_pnl(self) -> float:
        return self.portfolio.total_pnl

    @property
    def equity(self) -> float:
        """Current total equity = cash + Σ (quantity × avg_entry_price).

        For an accurate mark-to-market equity use ``portfolio.mark_to_market``
        after the latest bars are known.
        """
        return self.portfolio.cash + sum(
            p.quantity * p.avg_entry_price for p in self.portfolio.positions.values()
        )

    def reset(self) -> None:
        """Reset portfolio, trade history, and risk state."""
        self.portfolio = Portfolio(cash=self.capital or 0.0)
        self.trades = []
        self.risk.reset()
        self._last_date = None
