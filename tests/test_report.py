"""Tests for report.py: trades ledger + daily markdown report."""

import pytest

from alftrade.portfolio import Trade, CostModel, Portfolio
from alftrade.report import daily_report, load_ledger, write_ledger
from alftrade.broker import OrderRecord


def _fills():
    return [
        Trade(20240101, "600000.SH", 10.5, 200, "OPEN", cost=5.53, reason="REBALANCE"),
        Trade(20240101, "000001.SZ", 15.2, -100, "CLOSE", cost=6.0, pnl=120.0,
              reason="STOP_LOSS"),
    ]


def test_ledger_write_and_load(tmp_path):
    path = str(tmp_path / "trades.csv")
    n = write_ledger(path, _fills())
    assert n == 2
    rows = load_ledger(path)
    assert len(rows) == 2
    assert rows[0]["symbol"] == "600000.SH"
    assert rows[0]["side"] == "BUY"
    assert rows[0]["qty"] == "200"
    assert rows[1]["side"] == "SELL"


def test_ledger_append_is_idempotent(tmp_path):
    path = str(tmp_path / "trades.csv")
    write_ledger(path, _fills())
    # Re-writing the same fills must not duplicate rows.
    assert write_ledger(path, _fills()) == 0
    assert write_ledger(path, _fills()[:1]) == 0
    assert len(load_ledger(path)) == 2
    # New fills append.
    new = [Trade(20240102, "000002.SZ", 8.0, 300, "OPEN", cost=6.0)]
    assert write_ledger(path, new) == 1
    assert len(load_ledger(path)) == 3


def test_fee_breakdown_matches_cost():
    cm = CostModel()
    t = Trade(20240101, "A", 100.0, 500, "SELL")
    fees = cm.fee_breakdown("SELL", t.price, abs(t.quantity))
    total = fees["commission"] + fees["transfer"] + fees["stamp"]
    assert total == pytest.approx(cm.cost("SELL", t.price, abs(t.quantity)))
    # Sell pays stamp duty; a buy does not.
    fees_buy = cm.fee_breakdown("BUY", t.price, abs(t.quantity))
    assert fees_buy["stamp"] == 0.0


def test_daily_report_sections_and_numbers():
    fills = _fills()
    records = [
        OrderRecord(20240101, "600000.SH", "BUY", 200, 200, 10.5, "FILLED", "filled"),
        OrderRecord(20240101, "000001.SZ", "SELL", -100, -100, 15.2, "FILLED", "filled"),
        OrderRecord(20240101, "002353.SZ", "BUY", 1000, 0, 0.0, "REJECTED",
                    "limit-up at open"),
    ]
    pf = Portfolio(cash=800_000.0, initial_capital=1_000_000.0)
    from alftrade.data import DailyBar
    bars = {
        "600000.SH": DailyBar(20240101, "600000.SH", 10.5, 11.0, 11.2, 10.8,
                              1e6, 11e6, 10.0, is_suspended=False),
        "000001.SZ": DailyBar(20240101, "000001.SZ", 15.2, 15.5, 15.8, 15.0,
                              1e6, 15e6, 15.0),
    }
    text = daily_report(20240101, fills, records, pf, bars,
                        prev_equity=1_000_000.0)
    assert "# Daily report 20240101" in text
    assert "## Account" in text and "Equity" in text
    assert "600000.SH" in text and "BUY" in text
    assert "REJECTED" in text and "limit-up at open" in text
    assert "## Fees" in text and "Commission" in text
    assert "## Positions" in text


def test_daily_report_empty_day():
    pf = Portfolio(cash=1_000_000.0)
    text = daily_report(20240101, [], [], pf, {}, prev_equity=1_000_000.0)
    assert "No fills today." in text
    assert "All orders filled." in text
    assert "No open positions." in text