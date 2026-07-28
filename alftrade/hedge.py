"""
hedge.py - Simulated short-index-futures hedge for portfolio market exposure.

Models a dynamic short position in CFFEX index futures (IF / IC / IM) that
offsets the portfolio's market beta. The hedge is "simulated" — no actual
margin / exchange calls — but tracks all the economics faithfully:

  - rolling beta estimation (time-series OLS, 60-day default window)
  - daily mark-to-market settlement on the short position
  - dynamic lot rebalancing (only when beta drifts beyond a threshold)
  - contract roll cost (annualised contango / backwardation drag)

Two usage patterns:

  1. **Offline**: Instantiate outside the backtest loop, call
     ``step()`` once per day with portfolio return, market return,
     and index level.  The manager accumulates PnL and beta
     internally — you get a hedged-equity series you can compare
     with the raw one.

  2. **Inside ``PortfolioManager``**: Wire ``step()`` into
     ``decide()`` after ``mark_to_market`` to make the hedge part
     of the live simulation. (Future extension.)

Data flow per day:

    yesterday's index_price                    (stored internally)
    today's   index_price                     (supplied via step())
      → point_change = index_price − yesterday
      → settlement_pnl = −lots × point_change × multiplier

    portfolio_return, market_return
      → update rolling OLS beta
      → decide lots = floor(β × equity / notional_per_lot)
      → if β is not statistically significant, close the hedge
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class FuturesContract:
    """A CFFEX index-futures contract definition.

    Attributes:
        symbol: Exchange ticker (e.g. "IM", "IF", "IC").
        multiplier: Yuan per index point. CFFEX contracts are 200×.
        roll_cost_annual_bps: Annualised carry cost of the hedge
            (the "roll yield" drag). For IC / IM this is typically
            4–10 % p.a. due to persistent backwardation. Expressed
            as a decimal (0.06 = 6 %).
    """
    symbol: str
    multiplier: float = 200.0
    roll_cost_annual_bps: float = 0.06


class HedgeManager:
    """Simulated short-index-futures hedge.

    Maintains a rolling OLS beta estimate (portfolio returns on market
    returns) and computes the number of short futures lots needed to
    neutralise the market exposure.  Daily settlement adjusts the
    accumulated futures PnL.

    Args:
        contract: ``FuturesContract`` defining multiplier and roll cost.
        beta_window: Rolling days for beta estimation (default 60).
        min_beta_t: Minimum absolute t‑statistic to engage the hedge.
            Below this threshold the hedge is closed because the market
            exposure is not statistically reliable.
        max_leverage: Cap on the short notional relative to portfolio
            equity (default 1.5× — the system won't over‑hedge).
        adjust_threshold: Only rebalance lots when the estimated beta
            changes by more than this fraction (default 0.20).
    """

    def __init__(
        self,
        contract: Optional[FuturesContract] = None,
        beta_window: int = 60,
        min_beta_t: float = 2.0,
        max_leverage: float = 1.5,
        adjust_threshold: float = 0.20,
    ):
        self.contract = contract or FuturesContract("IM")
        self.beta_window = beta_window
        self.min_beta_t = min_beta_t
        self.max_leverage = max_leverage
        self.adjust_threshold = adjust_threshold

        self._port_rets: deque[float] = deque(maxlen=beta_window)
        self._mkt_rets: deque[float] = deque(maxlen=beta_window)
        self._beta: float = 1.0
        self._beta_t: float = 0.0
        self._lots: int = 0
        self._last_hedge_beta: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._index_price: float = 0.0

    # ── public API ──────────────────────────────────────────────────

    def step(
        self,
        date: int,
        equity: float,
        portfolio_return: float,
        market_return: float,
        index_price: float,
    ) -> tuple[float, int, float]:
        """Advance one day: settle → update beta → rebalance lots.

        Args:
            date: Trading-day ordinal (unused; kept for future roll logic).
            equity: Current portfolio total equity (yuan).
            portfolio_return: Today's raw portfolio return (fraction).
            market_return: Today's index futures return (fraction).
            index_price: Today's index level (points).

        Returns:
            (settlement_pnl, new_lots, beta)
                settlement_pnl — today's hedge PnL in yuan.
                new_lots       — number of short lots after rebalance.
                beta           — latest rolling OLS beta.
        """
        # 1. Daily mark‑to‑market on yesterday's open position.
        pnl = 0.0
        if self._lots > 0 and self._index_price > 0:
            point_change = index_price - self._index_price
            pnl = -self._lots * point_change * self.contract.multiplier

        # 2. Update rolling beta estimate.
        self._port_rets.append(portfolio_return)
        self._mkt_rets.append(market_return)
        self._update_beta()

        # 3. Rebalance lots if beta has drifted.
        old_lots = self._lots
        if self.should_hedge:
            notional = self._beta * equity
            notional = min(notional, self.max_leverage * equity)
            target_lots = max(0, int(notional / (index_price * self.contract.multiplier)))
            # Only trade if beta moved far enough or lots changed.
            if (self._lots == 0
                    or abs(self._beta - self._last_hedge_beta) / max(abs(self._last_hedge_beta), 0.01) > self.adjust_threshold
                    or target_lots != self._lots):
                self._lots = target_lots
                self._last_hedge_beta = self._beta
        else:
            self._lots = 0

        self._index_price = index_price
        self._cumulative_pnl += pnl

        # Apply daily roll-cost drag as a fraction of the hedge notional.
        if self._lots > 0:
            daily_roll = (self.contract.roll_cost_annual_bps / 252
                          * self._lots * index_price * self.contract.multiplier)
            self._cumulative_pnl -= daily_roll

        return pnl, self._lots, self._beta

    @property
    def should_hedge(self) -> bool:
        """True when the estimated beta is statistically reliable."""
        return abs(self._beta_t) >= self.min_beta_t

    @property
    def beta(self) -> float:
        """Latest rolling OLS beta coefficient."""
        return self._beta

    @property
    def beta_t(self) -> float:
        """Latest t‑statistic of the beta estimate."""
        return self._beta_t

    @property
    def lots(self) -> int:
        """Current number of short contracts."""
        return self._lots

    @property
    def cumulative_pnl(self) -> float:
        """Accumulated futures PnL since inception (yuan)."""
        return self._cumulative_pnl

    def reset(self) -> None:
        """Clear all state (start a new backtest)."""
        self._port_rets.clear()
        self._mkt_rets.clear()
        self._beta = 1.0
        self._beta_t = 0.0
        self._lots = 0
        self._cumulative_pnl = 0.0
        self._index_price = 0.0
        self._last_hedge_beta = 0.0

    # ── internal ────────────────────────────────────────────────────

    def _update_beta(self) -> None:
        """Rolling OLS beta with t‑statistic.

        Regresses portfolio returns on market returns over the
        available data in the window.  Falls back to β = 1.0 when
        there aren't enough points.
        """
        n = min(len(self._port_rets), len(self._mkt_rets))
        if n < 12:
            self._beta = 1.0
            self._beta_t = 0.0
            return

        r_p = list(self._port_rets)[-n:]
        r_m = list(self._mkt_rets)[-n:]

        mean_p = sum(r_p) / n
        mean_m = sum(r_m) / n

        cov = sum((a - mean_p) * (b - mean_m) for a, b in zip(r_p, r_m)) / n
        var_m = sum((b - mean_m) ** 2 for b in r_m) / n
        var_p = sum((a - mean_p) ** 2 for a in r_p) / n

        if var_m < 1e-12:
            self._beta = 1.0
            self._beta_t = 0.0
            return

        self._beta = cov / var_m

        # Standard error of the OLS slope:  se = sqrt( (var_p/var_m − β²) / (n−2) )
        if n > 2 and var_p > 1e-12:
            se2 = (var_p / var_m - self._beta ** 2) / (n - 2)
            se = max(se2, 0.0) ** 0.5
            self._beta_t = self._beta / se if se > 1e-12 else 0.0
        else:
            self._beta_t = 0.0


def compute_hedged_equity(
    equity_series: list[float],
    market_returns: list[float],
    hedge_pnl_series: list[float],
) -> list[float]:
    """Post‑process raw equity with accumulated hedge PnL.

    Returns the equity curve that *would have been* realised if
    each day's hedge settlement were added to cash, compounded
    forward.
    """
    if not equity_series:
        return []
    hedged = [equity_series[0] + hedge_pnl_series[0]]
    for t in range(1, len(equity_series)):
        hedged.append(equity_series[t] + sum(hedge_pnl_series[: t + 1]))
    return hedged
