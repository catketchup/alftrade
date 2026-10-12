"""Tests for paper.py: bootstrap, catch-up replay, state persistence, the
append-only ledger, and daily reports."""

import json
import os

import pytest

from alftrade import bars_to_frame
from alftrade.paper import PaperConfig, PaperTrader, ReplayUpdater, main
from alftrade.risk import RiskConfig


@pytest.fixture(scope="module")
def panel(market):
    bars_by_date, _, _ = market
    return bars_to_frame(bars_by_date)


@pytest.fixture(scope="module")
def pdates(panel):
    return sorted(panel["date"].unique().to_list())


def _cfg(state_dir, **kw):
    base = dict(capital=1e6, n_holdings=5, state_dir=str(state_dir),
                warmup_days=20, refit_days=1000)
    base.update(kw)
    return PaperConfig(**base)


def _paper(cfg, panel, max_date=None):
    return PaperTrader(cfg, ReplayUpdater(panel, max_date=max_date))


def test_first_run_bootstraps_then_trades(panel, pdates, tmp_path):
    s = _paper(_cfg(tmp_path), panel).run()
    assert s["processed_days"] == len(pdates) - 20   # warm-up days stay flat
    assert s["fills"] > 0
    # artifacts: state file, ledger, one report per processed day
    assert os.path.exists(os.path.join(str(tmp_path), "state.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "trades.csv"))
    reports = os.listdir(os.path.join(str(tmp_path), "reports"))
    assert len(reports) == s["processed_days"]
    with open(os.path.join(str(tmp_path), "reports", f"{pdates[20]}.md")) as f:
        assert "# Daily report" in f.read()


def test_second_run_is_noop(panel, pdates, tmp_path):
    pt = _paper(_cfg(tmp_path), panel)
    s1 = pt.run()
    ledger = os.path.join(str(tmp_path), "trades.csv")
    n_rows1 = sum(1 for _ in open(ledger))
    state1 = json.load(open(os.path.join(str(tmp_path), "state.json")))
    s2 = pt.run()
    assert s2["processed_days"] == 0
    assert s2["fills"] == 0
    assert sum(1 for _ in open(ledger)) == n_rows1
    state2 = json.load(open(os.path.join(str(tmp_path), "state.json")))
    assert state2["last_date"] == state1["last_date"]
    assert state2["pm"]["portfolio"]["cash"] == state1["pm"]["portfolio"]["cash"]


def test_catchup_matches_full_run(panel, pdates, tmp_path):
    """Partial run + later catch-up == a single full run (same equity)."""
    mid = pdates[60]
    pt = _paper(_cfg(tmp_path), panel, max_date=mid)
    s1 = pt.run()
    assert s1["processed_days"] == pdates.index(mid) - 20 + 1
    assert pt.run()["processed_days"] == 0            # no new data yet
    pt.updater.max_date = None                        # reveal the future
    s3 = pt.run()
    assert s3["processed_days"] == len(pdates) - pdates.index(mid) - 1

    ref = _paper(_cfg(tmp_path / "ref"), panel)
    s_ref = ref.run()
    assert s3["equity"] == pytest.approx(s_ref["equity"])


def test_pending_orders_survive_restart(panel, pdates, tmp_path):
    """Orders planned at the last close of run 1 execute at run 2's open."""
    mid = pdates[40]
    pt = _paper(_cfg(tmp_path), panel, max_date=mid)
    s1 = pt.run()
    assert s1["pending"] > 0 or s1["processed_days"] == 0
    if s1["pending"] == 0:
        pytest.skip("no pending orders at run end")
    pt.updater.max_date = None
    s2 = pt.run()
    assert s2["fills"] >= s1["pending"] - 0  # every pending order attempted
    assert s2["fills"] > 0


def test_halt_restored_from_state_blocks_new_opens(panel, pdates, tmp_path):
    """A halted account stays halted across runs; the circuit breaker then
    rejects new opens (only closes execute)."""
    mid = pdates[50]
    pt = _paper(_cfg(tmp_path), panel, max_date=mid)
    pt.run()
    # Simulate a triggered halt (e.g. daily-loss limit) by editing the state.
    path = os.path.join(str(tmp_path), "state.json")
    state = json.load(open(path))
    state["pm"]["risk"]["halted"] = True
    state["pm"]["risk"]["peak_equity"] = 5_000_000.0
    json.dump(state, open(path, "w"))

    pt.updater.max_date = None
    s = pt.run()
    assert s["halted"] is True
    assert s["processed_days"] == len(pdates) - pdates.index(mid) - 1
    # After the halt no new buys execute: every run-2 fill must be a sell.
    state2 = json.load(open(path))
    run2_trades = [t for t in state2["pm"]["trades"] if t["date"] > mid]
    assert run2_trades
    for t in run2_trades:
        assert t["quantity"] <= 0


def test_ledger_rows_are_stable_and_append_only(panel, pdates, tmp_path):
    pt = _paper(_cfg(tmp_path), panel)
    pt.run()
    ledger = os.path.join(str(tmp_path), "trades.csv")
    rows1 = [r for r in open(ledger)]
    header = rows1[0]
    assert "date,symbol,side" in header
    pt.run()                      # no-op: rows must not change
    assert [r for r in open(ledger)] == rows1


def test_refit_refreshes_calibration(panel, pdates, tmp_path):
    cfg = _cfg(tmp_path, refit_days=5)
    mid = pdates[60]
    pt = _paper(cfg, panel, max_date=mid)
    pt.run()
    state1 = json.load(open(os.path.join(str(tmp_path), "state.json")))
    pt.updater.max_date = None
    pt.run()
    state2 = json.load(open(os.path.join(str(tmp_path), "state.json")))
    # The calibration was refit with more history than the bootstrap fit
    # (proves refit actually happened), and the counter never exceeds the
    # refit interval.
    assert state2["calibration"]["n_dates"] > state1["calibration"]["n_dates"]
    assert state2["days_since_fit"] <= cfg.refit_days


def test_show_cron_prints_line(capsys):
    assert main(["--show-cron", "--state-dir", "x", "--pool", "csi300:30"]) == 0
    out = capsys.readouterr().out
    assert "python -m alftrade.paper" in out
    assert "csi300:30" in out


def test_show_cron_includes_export_dir(capsys):
    assert main(["--show-cron", "--state-dir", "x", "--pool", "csi300:30",
                 "--export-dir", "data/orders"]) == 0
    out = capsys.readouterr().out
    assert "--export-dir data/orders" in out


def test_export_dir_writes_ptrade_order_files(panel, pdates, tmp_path):
    """Shadow-book mode: the close-time plan of each day is exported, and the
    very last day (no next trading day) is skipped."""
    export_dir = tmp_path / "orders"
    cfg = _cfg(tmp_path, export_dir=str(export_dir))
    pt = _paper(cfg, panel)
    s = pt.run()
    exported = [d for d in os.listdir(str(export_dir))]
    # every processed day except the last has a next trading day
    assert len(exported) == max(s["processed_days"] - 1, 0)
    if exported:
        day_dir = os.path.join(str(export_dir), sorted(exported)[0])
        assert os.path.isfile(os.path.join(day_dir, "orders.csv"))
        assert os.path.isfile(os.path.join(day_dir, "orders.json"))
        assert os.path.isfile(os.path.join(day_dir, "target_positions.csv"))
        with open(os.path.join(day_dir, "orders.csv"),
                  encoding="utf-8-sig") as f:
            header = f.readline()
        assert header.startswith("plan_date,exec_date,symbol")