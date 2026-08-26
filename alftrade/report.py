"""
report.py - Daily paper-trading report and append-only trades ledger.

Two artifacts for the paper-trading loop:

  * ``write_ledger`` / ``load_ledger`` — an append-only CSV of every executed
    trade (audit trail; also allows replay-verification of account state).
  * ``daily_report`` — a human-readable markdown summary of one trading day:
    fills, order journal (rejected / partial / skipped orders), fee detail,
    positions, and account equity.
"""

import csv
from typing import Optional

from alftrade.portfolio import CostModel, Portfolio, Trade

LEDGER_COLUMNS = [
    "date", "symbol", "side", "qty", "price",
    "commission", "transfer", "stamp", "cost", "pnl", "reason",
]


def _fee_row(cost_model: CostModel, t: Trade) -> dict:
    side = "BUY" if t.quantity > 0 else "SELL"
    fees = cost_model.fee_breakdown(side, t.price, abs(t.quantity))
    return {
        "date": t.date, "symbol": t.symbol, "side": side,
        "qty": int(t.quantity), "price": round(t.price, 2),
        "commission": round(fees["commission"], 2),
        "transfer": round(fees["transfer"], 4),
        "stamp": round(fees["stamp"], 2),
        "cost": round(t.cost, 2), "pnl": round(t.pnl, 2),
        "reason": t.reason,
    }


def write_ledger(path: str, trades: list[Trade],
                 cost_model: Optional[CostModel] = None) -> int:
    """Append one row per trade to the ledger CSV; return rows appended.

    The ledger is append-only: callers pass only *new* fills (e.g. today's).
    """
    cost_model = cost_model or CostModel()
    rows = [_fee_row(cost_model, t) for t in trades]
    if not rows:
        return 0
    new_file = False
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            existing = {tuple(r[c] for c in LEDGER_COLUMNS) for r in reader}
    except (OSError, csv.Error):
        existing = set()
        new_file = True
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new_file:
            writer.writeheader()
        appended = 0
        for row in rows:
            key = tuple(str(row[c]) for c in LEDGER_COLUMNS)
            if key in existing:
                continue  # idempotent: never duplicate an already-logged fill
            writer.writerow(row)
            appended += 1
    return appended


def load_ledger(path: str) -> list[dict]:
    """Read the full ledger as a list of row dicts."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def daily_report(
    date: int,
    fills: list[Trade],
    records: list,
    portfolio: Portfolio,
    bars: dict,
    cost_model: Optional[CostModel] = None,
    prev_equity: Optional[float] = None,
) -> str:
    """Build the markdown daily report for one trading day.

    Args:
        date: Trading day.
        fills: Fills executed at this day's open.
        records: Order journal from the Broker (every attempted order).
        portfolio: The account's ``Portfolio``.
        bars: ``{symbol: DailyBar}`` for valuation.
        cost_model: For fee detail (defaults to ``CostModel()``).
        prev_equity: Yesterday's equity (for the daily return); optional.
    """
    cost_model = cost_model or CostModel()
    equity = portfolio.mark_to_market(bars)
    ret = (equity / prev_equity - 1.0) if prev_equity else None

    lines = [f"# Daily report {date}", ""]

    # ── Account summary ──
    lines.append("## Account")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Equity | {_fmt(equity)} |")
    lines.append(f"| Cash | {_fmt(portfolio.cash)} |")
    if ret is not None:
        lines.append(f"| Day return | {ret * 100:+.2f}% |")
    lines.append(f"| Realized PnL (all-time) | {_fmt(sum(p.realized_pnl for p in portfolio.positions.values()))} |")
    lines.append("")

    # ── Fills ──
    lines.append("## Fills")
    if fills:
        lines.append("| Symbol | Side | Qty | Price | Cost | PnL | Reason |")
        lines.append("|--------|------|----:|------:|-----:|----:|--------|")
        for t in fills:
            side = "BUY" if t.quantity > 0 else "SELL"
            lines.append(f"| {t.symbol} | {side} | {abs(int(t.quantity))} | {t.price:.2f} | "
                         f"{t.cost:.2f} | {t.pnl:+.2f} | {t.reason} |")
    else:
        lines.append("*No fills today.*")
    lines.append("")

    # ── Order journal (unfilled attempts) ──
    unfilled = [r for r in records if r.status != "FILLED"]
    lines.append("## Order journal")
    if unfilled:
        lines.append("| Symbol | Side | Requested | Filled | Status | Reason |")
        lines.append("|--------|------|----------:|-------:|--------|--------|")
        for r in unfilled:
            lines.append(f"| {r.symbol} | {r.side} | {abs(int(r.requested_qty))} | "
                         f"{abs(int(r.filled_qty))} | {r.status} | {r.reason} |")
    else:
        lines.append("*All orders filled.*")
    lines.append("")

    # ── Fees ──
    lines.append("## Fees")
    fee_keys = ("commission", "transfer", "stamp")
    totals = {k: 0.0 for k in fee_keys}
    for t in fills:
        fees = cost_model.fee_breakdown("BUY" if t.quantity > 0 else "SELL",
                                        t.price, abs(t.quantity))
        for k in fee_keys:
            totals[k] += fees[k]
    lines.append("| Fee | Amount |")
    lines.append("|-----|-------:|")
    lines.append(f"| Commission | {totals['commission']:.2f} |")
    lines.append(f"| Transfer | {totals['transfer']:.4f} |")
    lines.append(f"| Stamp duty | {totals['stamp']:.2f} |")
    lines.append(f"| **Total** | **{sum(totals.values()):.2f}** |")
    lines.append("")

    # ── Positions ──
    lines.append("## Positions")
    open_pos = [(s, p) for s, p in portfolio.positions.items() if p.quantity > 0]
    if open_pos:
        lines.append("| Symbol | Qty | Avg entry | Last close | Unrealized PnL |")
        lines.append("|--------|----:|----------:|-----------:|---------------:|")
        for sym, pos in open_pos:
            bar = bars.get(sym)
            close = bar.close if bar is not None else pos.avg_entry_price
            lines.append(f"| {sym} | {int(pos.quantity)} | {pos.avg_entry_price:.2f} | "
                         f"{close:.2f} | {pos.unrealized_pnl:+,.0f} |")
    else:
        lines.append("*No open positions.*")
    lines.append("")
    return "\n".join(lines)