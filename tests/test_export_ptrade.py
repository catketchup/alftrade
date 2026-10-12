"""Tests for the ptrade order export (format, rounding, idempotence)."""

import csv
import json
import os
from types import SimpleNamespace

import pytest

from alftrade.export_ptrade import export_ptrade_orders
from alftrade.portfolio import Portfolio, Trade


def _bar(close):
    return SimpleNamespace(close=close)


def _portfolio():
    p = Portfolio(cash=100_000.0, initial_capital=100_000.0)
    p.positions["600519"] = SimpleNamespace(quantity=500.0)
    p.positions["000858"] = SimpleNamespace(quantity=200.0)
    return p


def _orders():
    return [
        Trade(date=20260915, symbol="600519", price=1700.0, quantity=200,
              direction="INCREASE", reason="REBALANCE"),
        Trade(date=20260915, symbol="000858", price=50.0, quantity=-137,
              direction="REDUCE", reason="REBALANCE"),
    ]


def _bars():
    return {"600519": _bar(1700.0), "000858": _bar(50.0)}


def test_export_writes_all_three_files(tmp_path):
    out = str(tmp_path)
    path = export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                                _portfolio(), out_dir=out)
    assert path == os.path.join(out, "20260915")
    assert os.path.isfile(os.path.join(path, "orders.csv"))
    assert os.path.isfile(os.path.join(path, "orders.json"))
    assert os.path.isfile(os.path.join(path, "target_positions.csv"))


def test_orders_csv_columns_and_rows(tmp_path):
    out = str(tmp_path)
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out,
                         names={"600519": "贵州茅台", "000858": "五粮液"})
    with open(os.path.join(out, "20260915", "orders.csv"),
              encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "plan_date", "exec_date", "symbol", "name", "side", "quantity",
        "order_type", "ref_close", "est_notional", "note"]
    buy, sell = rows
    assert buy["symbol"] == "600519"
    assert buy["name"] == "贵州茅台"
    assert buy["side"] == "BUY" and buy["quantity"] == "200"
    assert buy["ref_close"] == "1700.0"
    assert buy["est_notional"] == "340000.0"
    assert sell["symbol"] == "000858"
    assert sell["side"] == "SELL" and sell["quantity"] == "137"


def test_buys_are_board_lots_sells_exact(tmp_path):
    """The exporter never touches quantities: buys arrive lot-rounded, sells
    may carry odd lots."""
    out = str(tmp_path)
    orders = [
        Trade(date=20260915, symbol="600519", price=1700.0, quantity=200,
              direction="INCREASE"),
        Trade(date=20260915, symbol="000858", price=50.0, quantity=-7,
              direction="CLOSE"),
    ]
    export_ptrade_orders(orders, 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    with open(os.path.join(out, "20260915", "orders.csv"),
              encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["quantity"] == "200"
    assert rows[1]["quantity"] == "7"


def test_target_positions_and_cash(tmp_path):
    out = str(tmp_path)
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    with open(os.path.join(out, "20260915", "target_positions.csv"),
              encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pos_rows = [r for r in rows if r["symbol"] != "__CASH__"]
    by_sym = {r["symbol"]: r for r in pos_rows}
    assert by_sym["600519"]["target_qty"] == "700"          # 500 + 200
    assert by_sym["000858"]["target_qty"] == "63"           # 200 - 137
    cash_row = [r for r in rows if r["symbol"] == "__CASH__"][0]
    # 100000 - 340000 + 6850 = -233150 (现金为负 = 需补足资金,对账可见)
    assert float(cash_row["target_qty"]) == pytest.approx(-233150.0)


def test_json_matches_csv_rows(tmp_path):
    out = str(tmp_path)
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    with open(os.path.join(out, "20260915", "orders.csv"),
              encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    meta = json.load(open(os.path.join(out, "20260915", "orders.json")))
    assert meta["plan_date"] == 20260915
    assert meta["exec_date"] == 20260916
    numeric = {"plan_date", "exec_date", "quantity"}
    numeric_floats = {"ref_close", "est_notional"}
    normalized = {
        k: (int(v) if k in numeric
            else float(v) if k in numeric_floats
            else v)
        for k, v in csv_rows[0].items()
    }
    assert meta["orders"][0] == normalized
    assert meta["targets"]["cash"] == pytest.approx(-233150.0)


def test_idempotent_overwrite(tmp_path):
    out = str(tmp_path)
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    first = open(os.path.join(out, "20260915", "orders.csv"),
                 encoding="utf-8-sig").read()
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    second = open(os.path.join(out, "20260915", "orders.csv"),
                  encoding="utf-8-sig").read()
    assert first == second


def test_utf8_bom_for_excel(tmp_path):
    out = str(tmp_path)
    export_ptrade_orders(_orders(), 20260915, 20260916, _bars(),
                         _portfolio(), out_dir=out)
    head = open(os.path.join(out, "20260915", "orders.csv"), "rb").read(3)
    assert head == b"\xef\xbb\xbf"


def test_no_orders_returns_none(tmp_path):
    out = str(tmp_path)
    assert export_ptrade_orders([], 20260915, 20260916, _bars(),
                                _portfolio(), out_dir=out) is None
    assert not os.path.exists(os.path.join(out, "20260915"))


def test_no_exec_date_returns_none(tmp_path):
    out = str(tmp_path)
    assert export_ptrade_orders(_orders(), 20260915, None, _bars(),
                                _portfolio(), out_dir=out) is None
    assert not os.path.exists(os.path.join(out, "20260915"))
