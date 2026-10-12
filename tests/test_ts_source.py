"""Tests for the optional Tushare-backed data source (mocked pro client)."""

import datetime

import pandas as pd
import polars as pl
import pytest

from alftrade.ts_source import (
    _load_token, _stock_basic_table, ts_industry_symbols, ts_valuation_table,
)


def _fake_basic_df():
    return pd.DataFrame({
        "ts_code": ["600519.SH", "000858.SZ", "600809.SH", "600559.SH",
                    "000001.SZ", "600888.SH"],
        "name": ["贵州茅台", "五粮液", "山西汾酒", "老白干酒", "平安银行",
                 "白酒股份"],
        "industry": ["酒、饮料和精制茶制造业"] * 4 + ["货币金融服务"] * 2,
        "market": ["主板"] * 6,
        "area": ["贵州", "四川", "山西", "河北", "深圳", "上海"],
    })


def _fake_pro():
    class FakePro:
        def stock_basic(self, exchange="", list_status="L", fields=None):
            return _fake_basic_df()
    return FakePro()


def test_stock_basic_table_writes_daily_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("alftrade.ts_source._pro", _fake_pro)
    df = _stock_basic_table(str(tmp_path))
    assert len(df) == 6
    assert "industry" in df.columns
    assert "crawl_date" not in df.columns
    cached = pl.read_parquet(tmp_path / "stock_basic.parquet")
    assert int(cached["crawl_date"].max()) == int(
        datetime.date.today().strftime("%Y%m%d"))


def test_stock_basic_table_cache_reuse(monkeypatch, tmp_path):
    """A same-day cache hit makes no second API call (throttle-friendly)."""
    calls = {"n": 0}

    def counting_pro():
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("stock_basic called twice")
        return _fake_pro()

    monkeypatch.setattr("alftrade.ts_source._pro", counting_pro)
    _stock_basic_table(str(tmp_path))
    df = _stock_basic_table(str(tmp_path))
    assert len(df) == 6
    assert calls["n"] == 1


def test_stock_basic_table_stale_fallback(monkeypatch, tmp_path, capsys):
    """A throttled call falls back to a stale cache."""
    import os
    cache_dir = str(tmp_path)
    stale = _fake_basic_df().iloc[:3].copy()
    t = pl.from_pandas(stale).with_columns([
        pl.col("ts_code").str.slice(0, 6).alias("code"),
        pl.lit(20200101).alias("crawl_date"),
    ]).select(["code", "name", "industry", "market", "area", "crawl_date"])
    os.makedirs(cache_dir, exist_ok=True)
    t.write_parquet(os.path.join(cache_dir, "stock_basic.parquet"))

    def failing_pro():
        raise Exception("抱歉，您访问接口(stock_basic)频率超限")

    monkeypatch.setattr("alftrade.ts_source._pro", failing_pro)
    df = _stock_basic_table(cache_dir)
    assert len(df) == 3
    assert "复用陈旧缓存" in capsys.readouterr().out


def test_ts_industry_symbols_industry_field(monkeypatch, tmp_path):
    monkeypatch.setattr("alftrade.ts_source._pro", _fake_pro)
    codes = ts_industry_symbols("酒", cache_dir=str(tmp_path))
    # sorted; 000001 excluded (货币金融), 600888 included via name 白酒股份
    assert codes == ["000858", "600519", "600559", "600809", "600888"]


def test_ts_industry_symbols_name_match(monkeypatch, tmp_path):
    monkeypatch.setattr("alftrade.ts_source._pro", _fake_pro)
    codes = ts_industry_symbols("白酒", cache_dir=str(tmp_path))
    assert codes == ["600888"]  # only 白酒股份's name contains 白酒


def test_ts_industry_symbols_no_match_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("alftrade.ts_source._pro", _fake_pro)
    with pytest.raises(ValueError, match="industry:酒"):
        ts_industry_symbols("半导体", cache_dir=str(tmp_path))


def test_ts_valuation_table_units_and_schema(monkeypatch, tmp_path):
    class FakePro:
        def trade_cal(self, exchange, start_date, end_date):
            return pd.DataFrame({
                "cal_date": ["20260910", "20260911"],
                "is_open": [1, 1],
            })

        def daily_basic(self, trade_date, fields):
            assert trade_date == "20260911"  # latest open day
            return pd.DataFrame({
                "ts_code": ["600519.SH", "000001.SZ"],
                "pe": [28.5, 5.2],
                "pe_ttm": [25.0, 5.0],
                "pb": [8.0, 0.7],
                "dv_ttm": [2.5, 3.0],
                "total_mv": [1.8e8, 2.5e7],  # 万元
                "circ_mv": [7.2e7, 2.0e7],
            })

        def stock_basic(self, exchange="", list_status="L", fields=None):
            return _fake_basic_df()

    monkeypatch.setattr("alftrade.ts_source._pro", lambda: FakePro())
    t = ts_valuation_table(cache_dir=str(tmp_path),
                           universe_cache_dir=str(tmp_path))
    assert t["code"].to_list() == ["600519", "000001"]
    assert t["cap"].to_list() == pytest.approx([18000.0, 2500.0])  # 亿元
    assert t["circ_cap"].to_list() == pytest.approx([7200.0, 2000.0])
    assert t["dy"].to_list() == [2.5, 3.0]
    assert t["name"].to_list() == ["贵州茅台", "平安银行"]
    assert int(t["crawl_date"].max()) == int(
        datetime.date.today().strftime("%Y%m%d"))


def test_load_token_env_wins(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tushare_token.txt").write_text("filetoken\n")
    monkeypatch.setenv("TUSHARE_TOKEN", "envtoken")
    assert _load_token(str(tmp_path)) == "envtoken"


def test_load_token_file_fallback(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tushare_token.txt").write_text("  filetoken  \n")
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    assert _load_token(str(tmp_path)) == "filetoken"


def test_load_token_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    assert _load_token(str(tmp_path)) == ""


def test_ts_valuation_table_without_token_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("alftrade.ts_source._load_token", lambda d=".": "")
    with pytest.raises(RuntimeError, match="data/tushare_token.txt"):
        ts_valuation_table(cache_dir=str(tmp_path))


def test_ts_valuation_table_names_optional(monkeypatch, tmp_path):
    """Missing stock_basic permission degrades names to null, not a crash."""
    class FakePro:
        def trade_cal(self, exchange, start_date, end_date):
            return pd.DataFrame({"cal_date": ["20260911"], "is_open": [1]})

        def daily_basic(self, trade_date, fields):
            return pd.DataFrame({
                "ts_code": ["600519.SH"], "pe": [28.5], "pe_ttm": [25.0],
                "pb": [8.0], "dv_ttm": [2.5], "total_mv": [1.8e8],
                "circ_mv": [7.2e7],
            })

        def stock_basic(self, **kw):
            raise Exception("无权限")

    monkeypatch.setattr("alftrade.ts_source._pro", lambda: FakePro())
    t = ts_valuation_table(cache_dir=str(tmp_path),
                           universe_cache_dir=str(tmp_path))
    assert t["name"].to_list() == [None]
    assert t["pe"].to_list() == [28.5]
