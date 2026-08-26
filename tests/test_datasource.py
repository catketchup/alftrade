"""Tests for datasource.py: cached simulated parquet, the unified switch, and
source-agnostic downstream."""

import os
import polars as pl
import pytest

from alftrade import (
    load_simulated, load_real, load_panel, DEFAULT_SIMULATED_ROOT,
    compute_factor_panel, factor_panel_to_dict, frame_to_bars, TradingEngine,
    FACTOR_NAMES,
)
from alftrade.schema import BAR_COLUMNS


def _write_daily_real(dirpath, rows):
    """Write a daily-bar parquet-by-date dir from rows.

    ``rows`` are dicts with keys ``date`` (str "YYYY-MM-DD"), ``symbol``, and
    daily OHLCV.  The symbol column is renamed to ``stock_code`` (the real
    A-share vendor convention) and a ``datetime`` column is added, then one
    file is written per date.
    """
    os.makedirs(dirpath, exist_ok=True)
    df = (pl.DataFrame(rows)
          .rename({"symbol": "stock_code"})
          .with_columns(pl.col("date").str.to_datetime("%Y-%m-%d").alias("datetime")))
    for (date_str,), day in df.group_by("date", maintain_order=True):
        day.select(["datetime", "stock_code", "open", "high", "low", "close",
                    "volume", "amount"]).write_parquet(
            os.path.join(dirpath, f"{date_str}.parquet"))


# ── load_simulated: caching + notice ───────────────────────────────────

def test_load_simulated_generates_parquet_cache(tmp_path, capsys):
    root = str(tmp_path / "sim")
    df = load_simulated(root=root, n_stocks=6, n_days=50, seed=1)

    assert set(df.columns) == set(BAR_COLUMNS)
    marketdata = os.path.join(root, "marketdata")
    assert os.path.isdir(marketdata)
    assert len(os.listdir(marketdata)) == 50
    assert os.path.isfile(os.path.join(root, "index.parquet"))

    out = capsys.readouterr().out
    assert "not found" in out  # notice issued on first generation


def test_load_simulated_reuses_cache(tmp_path, capsys):
    root = str(tmp_path / "sim")
    df1 = load_simulated(root=root, n_stocks=6, n_days=50, seed=1)
    capsys.readouterr()  # drain
    df2 = load_simulated(root=root, n_stocks=6, n_days=50, seed=1)

    assert df1.shape == df2.shape
    assert df1["close"].to_list() == df2["close"].to_list()
    out = capsys.readouterr().out
    assert "Using existing" in out


def test_load_simulated_force_regenerates(tmp_path):
    root = str(tmp_path / "sim")
    df1 = load_simulated(root=root, n_stocks=5, n_days=40, seed=1)
    df2 = load_simulated(root=root, n_stocks=5, n_days=40, seed=2, force=True)
    assert df1["close"].to_list() != df2["close"].to_list()


# ── load_real: daily-bar default ───────────────────────────────────────

def test_load_real_default_is_daily(tmp_path):
    _write_daily_real(str(tmp_path / "real"), [
        {"date": "2024-09-02", "symbol": "000001.SZ", "open": 10.0, "high": 10.2,
         "low": 9.9, "close": 10.1, "volume": 1000.0, "amount": 10100.0},
        {"date": "2024-09-02", "symbol": "600000.SH", "open": 8.0, "high": 8.1,
         "low": 7.9, "close": 8.0, "volume": 2000.0, "amount": 16000.0},
    ])
    df = load_real(str(tmp_path / "real"))  # intraday defaults to False
    assert set(df.columns) == set(BAR_COLUMNS)
    assert df.shape[0] == 2  # 2 symbols × 1 date


# ── load_panel: unified switch ─────────────────────────────────────────

def test_load_panel_simulated_and_real(tmp_path):
    sim_df = load_panel("simulated", root=str(tmp_path / "sim"),
                        n_stocks=6, n_days=50, seed=1)
    assert set(sim_df.columns) == set(BAR_COLUMNS)

    _write_daily_real(str(tmp_path / "real"), [
        {"date": "2024-09-02", "symbol": "000001.SZ", "open": 10.0, "high": 10.2,
         "low": 9.9, "close": 10.1, "volume": 1000.0, "amount": 10100.0},
    ])
    real_df = load_panel(str(tmp_path / "real"))
    assert set(real_df.columns) == set(BAR_COLUMNS)


def test_load_panel_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown data source"):
        load_panel("/nonexistent/path")


def test_load_panel_source_agnostic_pipeline(tmp_path):
    """Both sources feed an identical downstream: factors → engine."""
    sim_df = load_panel("simulated", root=str(tmp_path / "sim"),
                        n_stocks=8, n_days=60, seed=1)
    _write_daily_real(str(tmp_path / "real"), [
        {"date": "2024-09-02", "symbol": "000001.SZ", "open": 10.0, "high": 10.2,
         "low": 9.9, "close": 10.1, "volume": 1000.0, "amount": 10100.0},
        {"date": "2024-09-02", "symbol": "600000.SH", "open": 8.0, "high": 8.1,
         "low": 7.9, "close": 8.0, "volume": 2000.0, "amount": 16000.0},
    ])
    real_df = load_panel(str(tmp_path / "real"))

    for df in (sim_df, real_df):
        factors = factor_panel_to_dict(compute_factor_panel(df))
        assert set(factors)
        bars, _, _ = frame_to_bars(df)
        engine = TradingEngine(factors=factors)
        for d in sorted(bars):
            cs = engine.build_cross_section(d, bars[d])
            assert cs.z_matrix.shape[1] == len(FACTOR_NAMES)


def test_default_simulated_root_is_relative():
    assert DEFAULT_SIMULATED_ROOT == "data/simulated"
