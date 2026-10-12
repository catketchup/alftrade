"""
export_ptrade.py - Export next-day target orders for the ptrade trading system.

ptrade runs inside a broker-hosted environment, so alftrade cannot call its
API directly.  Instead the close-time plan (``Trader.pending``) is exported to
files that a ptrade-side strategy (see ``examples/ptrade_runner.py``) reads
every morning and executes at the open.

Exported per plan date into ``<out_dir>/<YYYYMMDD>/``:

  * ``orders.csv``         — one row per target order (utf-8-sig, Excel-ready)
  * ``orders.json``        — the same rows plus targets, machine-readable
  * ``target_positions.csv`` — the holdings the account should have after
    execution (last row ``__CASH__`` = target free cash)

Quantities are the exact values ptrade must send: buys are already rounded
down to board lots (100 shares) by the portfolio layer; sells are exact
(odd lots are sellable in one go).  The export is *idempotent*: re-exporting
the same plan date overwrites the files, so a re-run cannot double-place
orders as long as the ptrade side tracks executed plan dates.
"""

import csv
import datetime
import json
import os
from typing import Optional

from alftrade.portfolio import Portfolio, Trade

ORDERS_FILE = "orders.csv"
ORDERS_JSON = "orders.json"
TARGETS_FILE = "target_positions.csv"

ORDERS_COLUMNS = [
    "plan_date", "exec_date", "symbol", "name", "side", "quantity",
    "order_type", "ref_close", "est_notional", "note",
]


def _order_rows(
    orders: list[Trade],
    plan_date: int,
    exec_date: int,
    bars: dict,
    names: Optional[dict],
) -> list[dict]:
    rows = []
    for t in orders:
        ref = bars.get(t.symbol)
        ref_close = ref.close if ref is not None else t.price
        qty = abs(int(t.quantity))
        rows.append({
            "plan_date": plan_date,
            "exec_date": exec_date,
            "symbol": t.symbol,
            "name": (names or {}).get(t.symbol, ""),
            "side": "BUY" if t.quantity > 0 else "SELL",
            "quantity": qty,
            "order_type": "market",
            "ref_close": round(float(ref_close), 2),
            "est_notional": round(qty * float(ref_close), 2),
            "note": t.reason or "REBALANCE",
        })
    return rows


def _target_rows(
    portfolio: Portfolio,
    orders: list[Trade],
    bars: dict,
    names: Optional[dict],
) -> tuple[list[dict], float]:
    """Positions after applying the plan, plus the implied free cash.

    Cash delta is estimated at the plan-day close (``ref_close``); the real
    fills at tomorrow's open will differ slightly.
    """
    target: dict[str, float] = {s: p.quantity for s, p in portfolio.positions.items()}
    cash_delta = 0.0
    for t in orders:
        ref = bars.get(t.symbol)
        px = ref.close if ref is not None else t.price
        target[t.symbol] = target.get(t.symbol, 0.0) + t.quantity
        cash_delta -= t.quantity * px  # buys are negative → cash decreases

    rows = []
    for s, q in sorted(target.items()):
        if q <= 0:
            continue
        ref = bars.get(s)
        px = ref.close if ref is not None else 0.0
        rows.append({
            "symbol": s,
            "name": (names or {}).get(s, ""),
            "target_qty": int(q),
            "est_value": round(q * float(px), 2),
        })
    return rows, round(portfolio.cash + cash_delta, 2)


def export_ptrade_orders(
    orders: list[Trade],
    plan_date: int,
    exec_date: Optional[int],
    bars: dict,
    portfolio: Portfolio,
    out_dir: str = "data/orders",
    names: Optional[dict] = None,
) -> Optional[str]:
    """Write the ptrade order files for one plan date.

    Returns the plan directory path, or ``None`` when there is nothing to
    export (no orders, or no next trading day to execute on).  Same-day
    re-runs overwrite the files (idempotent).
    """
    if not orders or exec_date is None:
        return None
    rows = _order_rows(orders, plan_date, exec_date, bars, names)
    targets, target_cash = _target_rows(portfolio, orders, bars, names)
    day_dir = os.path.join(out_dir, str(plan_date))
    os.makedirs(day_dir, exist_ok=True)

    with open(os.path.join(day_dir, ORDERS_FILE), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ORDERS_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    target_fieldnames = ["symbol", "name", "target_qty", "est_value"]
    with open(os.path.join(day_dir, TARGETS_FILE), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=target_fieldnames)
        w.writeheader()
        w.writerows(targets)
        w.writerow({"symbol": "__CASH__", "name": "",
                    "target_qty": target_cash, "est_value": target_cash})

    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "plan_date": plan_date,
        "exec_date": exec_date,
        "orders": rows,
        "targets": {"cash": target_cash, "positions": targets},
    }
    with open(os.path.join(day_dir, ORDERS_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return day_dir
