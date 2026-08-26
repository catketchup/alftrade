"""
paper.py - Paper-trading loop: simulate real trading day by day, no API.

Bridges the offline backtest (demo.ipynb) and a live account:

  * **Data**: each run fetches the latest real daily bars (akshare, cached to
    parquet) and advances the account by every new trading day.  Missed days
    are replayed deterministically from their historical opens (catch-up).
  * **State**: the account (cash, positions, pending orders, risk state) and
    the factor calibration persist in ``state_dir`` across runs, so the
    strategy resumes exactly where it stopped.
  * **Artifacts**: an append-only trades ledger (``trades.csv``) and one
    markdown daily report per trading day (``reports/YYYYMMDD.md``).
  * **Model upkeep**: the Fama-MacBeth calibration refits automatically every
    ``refit_days`` new trading days, using only data available at that time.

Usage::

    python -m alftrade.paper --pool csi300:30 --capital 1000000
    python -m alftrade.paper --show-cron        # crontab line for daily runs
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import polars as pl

from alftrade.broker import Broker, ExecutionConfig
from alftrade.calibration import CalibrationResult, FactorCalibrator
from alftrade.data import frame_to_bars
from alftrade.datasource import fetch_daily_to_parquet, load_panel, resolve_symbols
from alftrade.engine import TradingEngine
from alftrade.factors import compute_factor_panel, factor_panel_to_dict
from alftrade.portfolio import CostModel, PortfolioManager, SignalAggregator
from alftrade.report import daily_report, write_ledger
from alftrade.risk import RiskConfig, RiskManager
from alftrade.trader import Trader

STATE_FILE = "state.json"
LEDGER_FILE = "trades.csv"
REPORTS_DIR = "reports"
HORIZON = 5


@dataclass
class PaperConfig:
    """Configuration for the paper-trading loop."""

    capital: float = 1_000_000.0
    n_holdings: int = 5
    pool: str = "csi300:30"
    state_dir: str = "paper_state"
    data_root: str = "data/real_daily"
    start_date: str = "20240101"
    warmup_days: int = 90
    refit_days: int = 30
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


class PanelUpdater:
    """Data-source abstraction: ``fetch`` returns the latest long panel."""

    def fetch(self) -> pl.DataFrame:
        raise NotImplementedError


class RealDataUpdater(PanelUpdater):
    """Fetches real daily bars (akshare) and caches them to parquet."""

    def __init__(self, root: str, symbols: list[str], start_date: str,
                 end_date: Optional[str] = None):
        self.root = root
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pl.DataFrame:
        end = self.end_date or _date.today().strftime("%Y%m%d")
        fetch_daily_to_parquet(self.symbols, self.start_date, end, self.root)
        return load_panel(self.root)


class ReplayUpdater(PanelUpdater):
    """Fake updater for tests: serves a fixed panel, optionally truncated to
    a maximum date (simulates time passing between runs)."""

    def __init__(self, panel: pl.DataFrame, max_date: Optional[int] = None):
        self.panel = panel
        self.max_date = max_date

    def fetch(self) -> pl.DataFrame:
        if self.max_date is None:
            return self.panel
        return self.panel.filter(pl.col("date") <= self.max_date)


def _fit(dates: list[int], bars, factors: dict) -> CalibrationResult:
    """Fit Fama-MacBeth on the given days, looking up forward returns only
    inside ``dates`` (no look-ahead into days beyond the fitting window)."""
    off = TradingEngine(factors=factors)
    cal = FactorCalibrator(horizon=HORIZON, min_samples=50, min_dates=10)
    bars_past = {d: bars[d] for d in dates}
    for d in dates:
        cal.collect(d, off.build_cross_section(d, bars[d]), bars_past)
    return cal.fit()


class PaperTrader:
    """Drives a ``Trader`` forward one run at a time with real/simulated data.

    Args:
        config: PaperConfig (capital, pool, risk, execution, ...).
        updater: PanelUpdater providing the latest panel.
    """

    def __init__(self, config: PaperConfig, updater: PanelUpdater):
        self.config = config
        self.updater = updater
        self.cost_model = CostModel()
        os.makedirs(config.state_dir, exist_ok=True)
        os.makedirs(os.path.join(config.state_dir, REPORTS_DIR), exist_ok=True)

    # ── paths ────────────────────────────────────────────────────────

    @property
    def _state_path(self) -> str:
        return os.path.join(self.config.state_dir, STATE_FILE)

    @property
    def _ledger_path(self) -> str:
        return os.path.join(self.config.state_dir, LEDGER_FILE)

    @property
    def _reports_dir(self) -> str:
        return os.path.join(self.config.state_dir, REPORTS_DIR)

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Fetch the latest panel and advance the account by every new day.

        Returns a summary dict: processed days, fills, journal statuses,
        final equity, halt flag, and the last processed date.
        """
        cfg = self.config
        panel = self.updater.fetch()
        dates = sorted(panel["date"].unique().to_list())
        bars, _uni, _mkt = frame_to_bars(panel)
        factors = factor_panel_to_dict(compute_factor_panel(panel))

        if os.path.exists(self._state_path):
            state = self._load_state()
            trader, cal = self._restore(state, factors)
            last_date = state["last_date"]
            prev_equity = state.get("prev_equity")
            days_since_fit = state.get("days_since_fit", 0)
        else:
            trader, cal, last_date = self._bootstrap(dates, bars, factors)
            prev_equity = None
            days_since_fit = 0

        new_days = [d for d in dates if d > last_date]
        n_fills = 0
        n_rejected = n_partial = n_skipped = 0
        for d in new_days:
            # Periodic refit on data available at day d only (never the future).
            if days_since_fit >= cfg.refit_days:
                fit_days = [x for x in dates if x <= d - HORIZON]
                if len(fit_days) >= 20:
                    cal = _fit(fit_days, bars, factors)
                    trader.engine = TradingEngine(factors=factors, calibration=cal)
                    days_since_fit = 0

            fills = trader.step(d, bars[d])
            n_fills += len(fills)
            day_records = [r for r in trader.order_log if r.date == d]
            for r in day_records:
                if r.status == "REJECTED":
                    n_rejected += 1
                elif r.status == "PARTIAL":
                    n_partial += 1
                elif r.status == "SKIPPED":
                    n_skipped += 1

            write_ledger(self._ledger_path, fills, self.cost_model)
            report = daily_report(
                d, fills, day_records, trader.portfolio.portfolio, bars[d],
                self.cost_model, prev_equity=prev_equity,
            )
            with open(os.path.join(self._reports_dir, f"{d}.md"), "w") as f:
                f.write(report)

            equity = trader.portfolio.portfolio.mark_to_market(bars[d])
            prev_equity = equity
            days_since_fit += 1
            self._save_state(trader, cal, d, prev_equity, days_since_fit)

        summary = {
            "processed_days": len(new_days),
            "fills": n_fills,
            "rejected": n_rejected,
            "partial": n_partial,
            "skipped": n_skipped,
            "equity": prev_equity,
            "halted": trader.portfolio.risk.halted,
            "pending": len(trader.pending),
            "last_date": dates[-1] if new_days else last_date,
        }
        self._print_summary(summary)
        return summary

    def _print_summary(self, s: dict) -> None:
        print("[paper] " + ", ".join([
            f"days={s['processed_days']}",
            f"fills={s['fills']}",
            f"rejected={s['rejected']}",
            f"partial={s['partial']}",
            f"skipped={s['skipped']}",
            f"equity={s['equity']:,.0f}" if s["equity"] is not None else "equity=n/a",
            f"halted={s['halted']}",
            f"pending={s['pending']}",
        ]))

    # ── bootstrap / restore ──────────────────────────────────────────

    def _new_trader(self, cal: CalibrationResult, factors: dict) -> Trader:
        engine = TradingEngine(factors=factors, calibration=cal)
        pm = PortfolioManager(
            capital=self.config.capital,
            n_holdings=self.config.n_holdings,
            weighting="equal",
            aggregator=SignalAggregator(calibration=cal),
            risk_manager=RiskManager(self.config.risk,
                                     initial_capital=self.config.capital),
            cost_model=self.cost_model,
        )
        return Trader(engine, pm, Broker(self.config.execution,
                                         cost_model=self.cost_model))

    def _bootstrap(self, dates: list[int], bars, factors: dict):
        """First run: calibrate on the warm-up window, then start trading
        from the warm-up's last day (no positions, no pending orders)."""
        cfg = self.config
        warm = dates[:cfg.warmup_days]
        cal = _fit(warm, bars, factors)
        trader = self._new_trader(cal, factors)
        return trader, cal, warm[-1]

    def _restore(self, state: dict, factors: dict):
        """Resume from a saved state file."""
        cal = CalibrationResult.from_state(state["calibration"])
        pm = PortfolioManager.from_state(
            state["pm"],
            aggregator=SignalAggregator(calibration=cal),
            cost_model=self.cost_model,
            risk_config=self.config.risk,
        )
        engine = TradingEngine(factors=factors, calibration=cal)
        trader = Trader.from_state(state["trader"], engine, pm)
        return trader, cal

    # ── persistence ──────────────────────────────────────────────────

    def _load_state(self) -> dict:
        with open(self._state_path) as f:
            return json.load(f)

    def _save_state(self, trader: Trader, cal: CalibrationResult, last_date: int,
                    prev_equity: float, days_since_fit: int) -> None:
        state = {
            "version": 1,
            "last_date": last_date,
            "prev_equity": prev_equity,
            "days_since_fit": days_since_fit,
            "calibration": cal.to_state(),
            "pm": trader.portfolio.to_state(),
            "trader": trader.to_state(),
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=1)


# ── CLI ────────────────────────────────────────────────────────────────


def _cron_line(state_dir: str, pool: str) -> str:
    cwd = os.getcwd()
    return (f"35 15 * * 1-5 cd {cwd} && python -m alftrade.paper "
            f"--pool {pool} --state-dir {state_dir} "
            f">> {state_dir}/paper.log 2>&1")


def _install_cron(line: str) -> None:
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True,
                                  text=True).stdout
    except OSError:
        print("crontab unavailable — add the line manually:\n" + line)
        return
    if line in existing:
        print("crontab entry already installed.")
        return
    try:
        subprocess.run(["crontab", "-"], input=existing + line + "\n", text=True,
                       check=True)
        print("Installed crontab entry:\n" + line)
    except subprocess.CalledProcessError:
        print("Failed to install crontab — add the line manually:\n" + line)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m alftrade.paper",
        description="Paper-trading loop: day-by-day simulation with real data.",
    )
    ap.add_argument("--pool", default="csi300:30",
                    help="Stock pool spec, e.g. csi300:30 / industry:白酒")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--n-holdings", type=int, default=5)
    ap.add_argument("--state-dir", default="paper_state")
    ap.add_argument("--data-root", default="data/real_daily")
    ap.add_argument("--start-date", default="20240101",
                    help="History fetch start (YYYYMMDD)")
    ap.add_argument("--end-date", default=None,
                    help="Latest fetch date (YYYYMMDD); default: today")
    ap.add_argument("--warmup-days", type=int, default=90)
    ap.add_argument("--refit-days", type=int, default=30)
    ap.add_argument("--timing", default="open",
                    choices=["open", "close_auction", "vwap"],
                    help="Execution timing (open auction / closing auction / "
                         "day VWAP). Can differ run-to-run; pending orders "
                         "execute under the run's timing.")
    ap.add_argument("--show-cron", action="store_true",
                    help="Print a crontab line for daily runs")
    ap.add_argument("--install-cron", action="store_true",
                    help="Add the daily crontab entry (workdays 15:35)")
    args = ap.parse_args(argv)

    if args.show_cron or args.install_cron:
        line = _cron_line(args.state_dir, args.pool)
        if args.install_cron:
            _install_cron(line)
        else:
            print(line)
        return 0

    symbols = resolve_symbols(args.pool)
    config = PaperConfig(
        capital=args.capital,
        n_holdings=args.n_holdings,
        pool=args.pool,
        state_dir=args.state_dir,
        data_root=args.data_root,
        start_date=args.start_date,
        warmup_days=args.warmup_days,
        refit_days=args.refit_days,
        execution=ExecutionConfig(execution_timing=args.timing),
    )
    updater = RealDataUpdater(args.data_root, symbols, args.start_date,
                              args.end_date)
    PaperTrader(config, updater).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())