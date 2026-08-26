"""Tests for ingest.py: intraday→daily aggregation + field derivation, and
data.frame_to_bars round-trip."""

import os
import polars as pl
import pytest

from alftrade.ingest import ingest_parquet, STANDARD_COLUMNS, limit_pct_for_symbol
from alftrade import frame_to_bars, bars_to_frame, generate_simulated_market


# ── Helpers ────────────────────────────────────────────────────────────

def _write_intraday(root, symbol="000001.SZ", date="20240902"):
    """Write one intraday parquet file with a few 1-min bars for one symbol."""
    rows = [
        {"datetime": f"{date} 09:31:00", "stock_code": symbol,
         "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
         "volume": 1000.0, "amount": 10100.0},
        {"datetime": f"{date} 10:00:00", "stock_code": symbol,
         "open": 10.1, "high": 10.5, "low": 10.0, "close": 10.4,
         "volume": 2000.0, "amount": 20500.0},
        {"datetime": f"{date} 15:00:00", "stock_code": symbol,
         "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5,
         "volume": 3000.0, "amount": 31300.0},
    ]
    pl.DataFrame(rows).with_columns(
        pl.col("datetime").str.to_datetime("%Y%m%d %H:%M:%S")
    ).write_parquet(os.path.join(root, f"{date}.parquet"))


# ── Aggregation ────────────────────────────────────────────────────────

def test_ingest_aggregates_intraday_to_daily(tmp_path):
    _write_intraday(str(tmp_path))
    df = ingest_parquet(str(tmp_path), intraday=True)
    assert df.columns == STANDARD_COLUMNS
    assert df.shape[0] == 1  # one symbol × one date
    row = df.row(0, named=True)
    # open = first bar's open, high = max, low = min, close = last bar's close
    assert row["open"] == pytest.approx(10.0)
    assert row["high"] == pytest.approx(10.6)
    assert row["low"] == pytest.approx(9.9)
    assert row["close"] == pytest.approx(10.5)
    assert row["volume"] == pytest.approx(1000 + 2000 + 3000)
    assert row["amount"] == pytest.approx(10100 + 20500 + 31300)


# ── Derivation ─────────────────────────────────────────────────────────

def test_ingest_derives_prev_close_across_dates(tmp_path):
    _write_intraday(str(tmp_path), date="20240902")
    # Day 2: same symbol, closes at 11.0 (prev day close = 10.5)
    rows = [
        {"datetime": "20240903 09:31:00", "stock_code": "000001.SZ",
         "open": 10.5, "high": 11.2, "low": 10.5, "close": 11.0,
         "volume": 1000.0, "amount": 11000.0},
        {"datetime": "20240903 15:00:00", "stock_code": "000001.SZ",
         "open": 11.0, "high": 11.1, "low": 10.9, "close": 11.0,
         "volume": 1000.0, "amount": 11000.0},
    ]
    pl.DataFrame(rows).with_columns(
        pl.col("datetime").str.to_datetime("%Y%m%d %H:%M:%S")
    ).write_parquet(os.path.join(str(tmp_path), "20240903.parquet"))

    df = ingest_parquet(str(tmp_path), intraday=True).sort("date")
    assert df["prev_close"].to_list() == pytest.approx([10.5, 10.5])
    # limit prices = prev_close × (1 ± 0.10) rounded to 0.01
    assert df["limit_up_price"][1] == pytest.approx(round(10.5 * 1.10, 2))
    assert df["limit_down_price"][1] == pytest.approx(round(10.5 * 0.90, 2))


def test_ingest_derives_market_return_equal_weight(tmp_path):
    # Day 1: A=100, B=100.  Day 2: A=110 (+10%), B=90 (-10%) → cross-sectional mean = 0.
    rows = []
    for date, sym, close in [
        ("2024-09-02", "000001.SZ", 100.0),
        ("2024-09-02", "600000.SH", 100.0),
        ("2024-09-03", "000001.SZ", 110.0),
        ("2024-09-03", "600000.SH", 90.0),
    ]:
        rows.append({"datetime": f"{date} 15:00:00", "stock_code": sym,
                     "open": close, "high": close, "low": close, "close": close,
                     "volume": 1000.0, "amount": close * 1000.0})
    df = pl.DataFrame(rows).with_columns(pl.col("datetime").str.to_datetime())

    # Write one file per date.
    for date, day in df.group_by(pl.col("datetime").dt.strftime("%Y%m%d")):
        date_str = date
        day.write_parquet(os.path.join(str(tmp_path), f"{date_str}.parquet"))

    out = ingest_parquet(str(tmp_path), intraday=True).sort(["date", "symbol"])
    mkt_day2 = out.filter(pl.col("date") == 20240903)["market_return"].unique().to_list()
    assert mkt_day2 == pytest.approx([0.0], abs=1e-6)


# ── frame_to_bars round-trip ───────────────────────────────────────────

def test_frame_to_bars_roundtrip():
    bars, universe, market_returns = generate_simulated_market(n_stocks=5, n_days=50, seed=1)
    long_df = bars_to_frame(bars)
    bars2, uni2, mkt2 = frame_to_bars(long_df)
    assert set(bars2.keys()) == set(bars.keys())
    assert set(uni2.symbols) == set(universe.symbols)
    assert set(mkt2.keys()) == set(market_returns.keys())
    d = sorted(bars)[-1]
    for sym in bars[d]:
        o, l = bars[d][sym], bars2[d][sym]
        assert l.open == o.open and l.close == o.close
        assert l.prev_close == o.prev_close
        assert l.market_return == o.market_return
        assert l.limit_up_price == o.limit_up_price
        assert l.limit_down_price == o.limit_down_price
        assert l.is_suspended == o.is_suspended
        assert l.total_shares == o.total_shares


# ── Board-specific price limits ────────────────────────────────────────

def test_limit_pct_for_symbol_boards():
    """Board-specific limits: 10% main, 20% STAR/ChiNext, 30% BSE."""
    assert limit_pct_for_symbol("600000.SH") == 0.10   # Shanghai main
    assert limit_pct_for_symbol("000001.SZ") == 0.10   # Shenzhen main
    assert limit_pct_for_symbol("002384.SZ") == 0.10   # SME board
    assert limit_pct_for_symbol("688072.SH") == 0.20   # STAR
    assert limit_pct_for_symbol("300251.SZ") == 0.20   # ChiNext
    assert limit_pct_for_symbol("301165.SZ") == 0.20   # ChiNext (301)
    assert limit_pct_for_symbol("830000.BJ") == 0.30   # BSE
    assert limit_pct_for_symbol("430047") == 0.30      # BSE bare code


def test_limit_pct_for_symbol_bare_code():
    """A bare 6-digit code works without an exchange suffix."""
    assert limit_pct_for_symbol("600000") == 0.10
    assert limit_pct_for_symbol("688072") == 0.20


def test_ingest_derives_board_specific_limits(tmp_path):
    """limit_up/down prices use the board-specific fraction per symbol."""
    rows = [
        {"date": "2024-09-02", "symbol": "600000.SH", "open": 10.0, "high": 10.2,
         "low": 9.9, "close": 10.0, "volume": 1000.0, "amount": 10000.0},
        {"date": "2024-09-02", "symbol": "688072.SH", "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": 1000.0, "amount": 100000.0},
        {"date": "2024-09-03", "symbol": "600000.SH", "open": 10.0, "high": 10.2,
         "low": 9.9, "close": 10.0, "volume": 1000.0, "amount": 10000.0},
        {"date": "2024-09-03", "symbol": "688072.SH", "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": 1000.0, "amount": 100000.0},
    ]
    os.makedirs(os.path.join(str(tmp_path)), exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d").alias("datetime")
    ).rename({"symbol": "stock_code"})
    for (date_str,), day in df.group_by("date", maintain_order=True):
        day.select(["datetime", "stock_code", "open", "high", "low", "close",
                    "volume", "amount"]).write_parquet(
            os.path.join(str(tmp_path), f"{date_str}.parquet"))

    out = ingest_parquet(str(tmp_path)).sort(["symbol", "date"])
    # 600000 (main board): prev_close=10 on day 2 → limit_up = 10*1.10 = 11.00
    m600 = out.filter(pl.col("symbol") == "600000.SH").sort("date")
    assert m600["limit_up_price"][1] == pytest.approx(round(10.0 * 1.10, 2))
    assert m600["limit_down_price"][1] == pytest.approx(round(10.0 * 0.90, 2))
    # 688072 (STAR): prev_close=100 on day 2 → limit_up = 100*1.20 = 120.00
    s688 = out.filter(pl.col("symbol") == "688072.SH").sort("date")
    assert s688["limit_up_price"][1] == pytest.approx(round(100.0 * 1.20, 2))
    assert s688["limit_down_price"][1] == pytest.approx(round(100.0 * 0.80, 2))
