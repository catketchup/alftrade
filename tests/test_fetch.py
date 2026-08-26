"""Tests for the akshare daily-fetch path (mocked, no network) and
fetch_daily_to_parquet caching."""

import os
import pandas as pd
import polars as pl
import pytest

from alftrade import (
    ingest_akshare, fetch_daily_to_parquet, load_panel, frame_to_bars,
)
from alftrade.schema import BAR_COLUMNS


# ── Mocks ──────────────────────────────────────────────────────────────

def _fake_akshare_daily(symbol, start_date, end_date, adjust=""):
    """Stand-in for akshare.stock_zh_a_daily returning Sina-style columns."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "open": [10.0] * n,
        "high": [10.5] * n,
        "low": [9.5] * n,
        "close": [10.0 + i * 0.1 for i in range(n)],
        "volume": [1000.0] * n,
        "amount": [10000.0] * n,
        "outstanding_share": [1e9] * n,
        "turnover": [0.01] * n,
    })


def _fake_bar_df():
    """A minimal long table in the canonical BAR_COLUMNS schema."""
    return pl.DataFrame({
        "date": [20240102, 20240102, 20240103, 20240103],
        "symbol": ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
        "open": [10.0] * 4, "high": [10.5] * 4, "low": [9.5] * 4, "close": [10.2] * 4,
        "volume": [1000.0] * 4, "amount": [10000.0] * 4,
        "prev_close": [10.0] * 4,
        "limit_up_price": [11.0] * 4, "limit_down_price": [9.0] * 4,
        "is_suspended": [False] * 4, "total_shares": [1e9] * 4,
        "market_return": [0.001] * 4,
    })


# ── ingest_akshare (mocked) ────────────────────────────────────────────

def test_ingest_akshare_mocked(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_daily", _fake_akshare_daily)
    df = ingest_akshare(["000001", "600000"], "20240101", "20240105")

    assert set(df.columns) == set(BAR_COLUMNS)
    # 2 symbols × 5 days = 10 rows.
    assert df.shape[0] == 10
    symbols = sorted(df["symbol"].unique().to_list())
    assert symbols == ["000001.SZ", "600000.SH"]
    # prev_close on day 2 of 000001 = day 1 close.
    first_close = df.filter(pl.col("symbol") == "000001.SZ").sort("date")["close"][0]
    second_prev = df.filter(pl.col("symbol") == "000001.SZ").sort("date")["prev_close"][1]
    assert second_prev == pytest.approx(first_close)
    # limit_up = prev_close * 1.10 rounded to 0.01.
    assert df["limit_up_price"][1] == pytest.approx(round(first_close * 1.10, 2))


def test_ingest_akshare_exchange_inference(monkeypatch):
    """6/9-prefixed codes map to .SH, others to .SZ."""
    seen = {}

    def _record(symbol, **kwargs):
        seen["symbol"] = symbol
        return _fake_akshare_daily(symbol, "", "")

    monkeypatch.setattr("akshare.stock_zh_a_daily", _record)
    ingest_akshare(["600000", "000001"], "20240101", "20240102")
    # Last recorded call is 000001 → sz prefix.
    assert seen["symbol"] == "sz000001"


# ── fetch_daily_to_parquet (caching) ───────────────────────────────────

def test_fetch_daily_to_parquet_caches(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("alftrade.datasource.ingest_akshare",
                        lambda *a, **k: _fake_bar_df())
    root = str(tmp_path / "real")
    symbols = ["000001", "600000"]  # match the 2 symbols in _fake_bar_df

    df1 = fetch_daily_to_parquet(symbols, "20240101", "20240105", root)
    assert os.path.isdir(os.path.join(root, "marketdata"))
    assert len(os.listdir(os.path.join(root, "marketdata"))) == 2  # 2 dates
    assert set(df1.columns) == set(BAR_COLUMNS)
    out1 = capsys.readouterr().out
    assert "Fetching" in out1

    # Second call reuses the cache (no re-fetch).
    df2 = fetch_daily_to_parquet(symbols, "20240101", "20240105", root)
    assert df1["close"].to_list() == df2["close"].to_list()
    out2 = capsys.readouterr().out
    assert "Using existing" in out2


def test_fetch_daily_to_parquet_force(tmp_path, capsys, monkeypatch):
    calls = {"n": 0}

    def _counted(*a, **k):
        calls["n"] += 1
        return _fake_bar_df()

    monkeypatch.setattr("alftrade.datasource.ingest_akshare", _counted)
    root = str(tmp_path / "real")
    symbols = ["000001", "600000"]
    fetch_daily_to_parquet(symbols, "20240101", "20240105", root)
    fetch_daily_to_parquet(symbols, "20240101", "20240105", root, force=True)
    assert calls["n"] == 2  # force triggers a second fetch


def test_fetch_daily_to_parquet_invalidates_on_pool_change(tmp_path, monkeypatch):
    """Changing the requested stock pool re-fetches instead of reusing the cache."""
    def _fake_bar_single():
        return pl.DataFrame({
            "date": [20240102, 20240103],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [10.0] * 2, "high": [10.5] * 2, "low": [9.5] * 2, "close": [10.2] * 2,
            "volume": [1000.0] * 2, "amount": [10000.0] * 2,
            "prev_close": [10.0] * 2,
            "limit_up_price": [11.0] * 2, "limit_down_price": [9.0] * 2,
            "is_suspended": [False] * 2, "total_shares": [1e9] * 2,
            "market_return": [0.001] * 2,
        })

    monkeypatch.setattr("alftrade.datasource.ingest_akshare", lambda *a, **k: _fake_bar_single())
    root = str(tmp_path / "real")
    fetch_daily_to_parquet(["000001"], "20240101", "20240105", root)
    # Now request a different pool → should re-fetch (cache has only 000001).
    from alftrade.datasource import _cache_matches
    assert not _cache_matches(root, ["600000"])


def test_fetch_then_load_panel(tmp_path, monkeypatch):
    """After fetch, load_panel(root) reads the cache without network."""
    monkeypatch.setattr("alftrade.datasource.ingest_akshare",
                        lambda *a, **k: _fake_bar_df())
    root = str(tmp_path / "real")
    fetch_daily_to_parquet(["000001", "600000"], "20240101", "20240105", root)

    df = load_panel(root)
    assert set(df.columns) == set(BAR_COLUMNS)
    bars, _, _ = frame_to_bars(df)
    assert len(bars) == 2


# ── Real-network smoke test (manual; skipped by default) ───────────────

@pytest.mark.skip(reason="requires network + akshare; run manually")
def test_ingest_akshare_real_network():
    df = ingest_akshare(["000001", "600000"], "20240101", "20240201")
    assert set(df.columns) == set(BAR_COLUMNS)
    assert df.shape[0] > 0
