"""Tests for stock-pool helpers: csi300_symbols, industry_symbols, resolve_symbols."""

import datetime

import pandas as pd
import polars as pl
import pytest

from alftrade import (
    csi300_symbols, industry_symbols, market_cap_symbols, resolve_symbols,
    pool_label, symbol_names,
)
from alftrade.datasource import _parse_valuation_spec, valuation_symbols


@pytest.fixture(autouse=True)
def _no_tushare_network(monkeypatch):
    """Keep universe tests offline: the tushare-first chain fails fast and
    the mocked EM/Sina paths below take over.  (A real token in
    ``data/tushare_token.txt`` must not leak into unit tests.)"""
    monkeypatch.setattr(
        "alftrade.datasource.ts_valuation_table",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("tushare disabled in tests")))


def _fake_index_cons(symbol="000300"):
    return pd.DataFrame({
        "品种代码": [f"{600000 + i}" for i in range(10)],
        "品种名称": [f"股票{i}" for i in range(10)],
        "纳入日期": ["2026-06-15"] * 10,
    })


def _fake_spot_em():
    return pd.DataFrame({
        "代码": ["600000", "000001", "300001", "688001"],
        "名称": ["浦发银行", "平安银行", "小盘A", "小盘B"],
        "总市值": [3000.0, 2000.0, 100.0, 50.0],
        "流通市值": [2500.0, 1800.0, 90.0, 40.0],
    })


def test_csi300_symbols_all(monkeypatch, tmp_path):
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons)
    codes = csi300_symbols(cache_dir=str(tmp_path))
    assert codes == [f"{600000 + i}" for i in range(10)]


def test_csi300_symbols_slice(monkeypatch, tmp_path):
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons)
    codes = csi300_symbols(3, cache_dir=str(tmp_path))
    assert codes == ["600000", "600001", "600002"]


def test_csi300_symbols_cache_reuse(monkeypatch, tmp_path):
    """A same-day cache hit needs no network call."""
    calls = {"n": 0}

    def _counting_index_cons(symbol="000300"):
        calls["n"] += 1
        return _fake_index_cons(symbol)

    monkeypatch.setattr("akshare.index_stock_cons", _counting_index_cons)
    assert len(csi300_symbols(cache_dir=str(tmp_path))) == 10
    assert len(csi300_symbols(cache_dir=str(tmp_path))) == 10
    assert calls["n"] == 1


def test_csi300_symbols_stale_fallback(monkeypatch, tmp_path, capsys):
    """A throttled Sina page reuses the stale cache."""
    import os
    cache_dir = str(tmp_path)
    os.makedirs(cache_dir, exist_ok=True)
    pl.DataFrame({
        "code": [f"{600000 + i}" for i in range(5)],
        "crawl_date": [20200101] * 5,
    }).write_parquet(os.path.join(cache_dir, "csi300.parquet"))

    def _failing_index_cons(symbol="000300"):
        raise AttributeError("'NoneType' object has no attribute 'find'")

    monkeypatch.setattr("akshare.index_stock_cons", _failing_index_cons)
    codes = csi300_symbols(cache_dir=cache_dir)
    assert len(codes) == 5
    assert "复用陈旧缓存" in capsys.readouterr().out


def test_csi300_symbols_no_cache_raises(monkeypatch, tmp_path):
    def _failing_index_cons(symbol="000300"):
        raise AttributeError("'NoneType' object has no attribute 'find'")

    monkeypatch.setattr("akshare.index_stock_cons", _failing_index_cons)
    with pytest.raises(RuntimeError, match="无法获取沪深300成分"):
        csi300_symbols(cache_dir=str(tmp_path))


def test_industry_symbols(monkeypatch):
    def _fake_ts_industry(keyword, cache_dir="data/universe"):
        assert keyword == "白酒"
        return ["600519", "000858", "000568"]

    monkeypatch.setattr("alftrade.datasource.ts_industry_symbols",
                        _fake_ts_industry)
    codes = industry_symbols("白酒")
    assert codes == ["600519", "000858", "000568"]


def test_resolve_symbols_list_passthrough():
    assert resolve_symbols(["600519", "000001"]) == ["600519", "000001"]
    assert resolve_symbols(("600519",)) == ["600519"]


def test_resolve_symbols_csi300(monkeypatch):
    def _fake_csi300(n=None, cache_dir="data/universe"):
        codes = [f"{600000 + i}" for i in range(10)]
        return codes[:n] if n is not None else codes

    monkeypatch.setattr("alftrade.datasource.csi300_symbols", _fake_csi300)
    assert resolve_symbols("csi300") == [f"{600000 + i}" for i in range(10)]
    assert resolve_symbols("csi300:2") == ["600000", "600001"]


def test_resolve_symbols_industry(monkeypatch):
    def _fake_ts_industry(keyword, cache_dir="data/universe"):
        assert keyword == "白酒"
        return ["600519", "000858", "000568"]

    monkeypatch.setattr("alftrade.datasource.ts_industry_symbols",
                        _fake_ts_industry)
    assert resolve_symbols("industry:白酒") == ["600519", "000858", "000568"]


def test_resolve_symbols_comma_string():
    assert resolve_symbols("600519,000001, 600036") == ["600519", "000001", "600036"]


def test_resolve_symbols_single_code():
    assert resolve_symbols("600519") == ["600519"]


def test_resolve_symbols_invalid():
    with pytest.raises(ValueError):
        resolve_symbols(12345)


# ── Market-cap stock pool ──────────────────────────────────────────────

def test_market_cap_symbols_top(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _fake_spot_em)
    assert market_cap_symbols(2) == ["600000", "000001"]  # top 2 by 总市值


def test_market_cap_symbols_small(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _fake_spot_em)
    # ascending: 688001(50) < 300001(100) < 000001(2000) < 600000(3000)
    assert market_cap_symbols(2, ascending=True) == ["688001", "300001"]


def test_market_cap_symbols_float_cap(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _fake_spot_em)
    assert market_cap_symbols(2, float_cap=True) == ["600000", "000001"]


def test_market_cap_symbols_tushare_primary(monkeypatch, tmp_path):
    """With tushare available, the pool ranks whole-market caps from
    ``daily_basic`` (circ_cap for float_cap) without touching EM/Sina."""
    import datetime
    table = pl.DataFrame({
        "code": ["600000", "000001", "300001", "688001"],
        "cap": [3000.0, 2000.0, 100.0, 50.0],
        "circ_cap": [900.0, 1800.0, 60.0, 40.0],
    }).with_columns(pl.lit(
        int(datetime.date.today().strftime("%Y%m%d"))).alias("crawl_date"))
    monkeypatch.setattr("alftrade.datasource.ts_valuation_table",
                        lambda cache_dir: table)
    monkeypatch.setattr(
        "alftrade.datasource.DEFAULT_VALUATION_CACHE_DIR", str(tmp_path))

    assert market_cap_symbols(2, cache_dir=str(tmp_path)) == ["600000", "000001"]
    assert market_cap_symbols(2, ascending=True,
                              cache_dir=str(tmp_path)) == ["688001", "300001"]
    # float_cap ranks by circ_cap: 000001(1800) > 600000(900)
    assert market_cap_symbols(2, float_cap=True,
                              cache_dir=str(tmp_path)) == ["000001", "600000"]
    # the tushare table is persisted so valuation: pools can reuse it
    cached = pl.read_parquet(tmp_path / "sw_valuation.parquet")
    assert sorted(cached["code"].to_list()) == sorted(table["code"].to_list())


def test_resolve_symbols_market_cap(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _fake_spot_em)
    assert resolve_symbols("market_cap:2") == ["600000", "000001"]
    assert resolve_symbols("market_cap_small:2") == ["688001", "300001"]


def test_market_cap_symbols_sina_fallback(monkeypatch, tmp_path, capsys):
    """When Eastmoney is down, fall back to Sina per-stock market cap over
    CSI300; results are cached per day so a second call needs no network."""
    calls = {"n": 0}

    def _failing_spot_em():
        raise ConnectionError("blocked")

    def _fake_index_cons_3(symbol="000300"):
        return pd.DataFrame({
            "品种代码": ["600000", "000001", "300001"],
            "品种名称": ["浦发银行", "平安银行", "小盘A"],
            "纳入日期": ["2026-06-15"] * 3,
        })

    # market cap = close × outstanding_share; craft so 600000 > 000001 > 300001
    def _fake_daily(symbol, start_date, end_date, adjust=""):
        calls["n"] += 1
        code = symbol[2:]
        shares = {"600000": 100e8, "000001": 50e8, "300001": 10e8}
        caps = {"600000": 3000e8, "000001": 2000e8, "300001": 100e8}
        price = caps[code] / shares[code]
        return pd.DataFrame({
            "close": [price, price],
            "outstanding_share": [shares[code], shares[code]],
        })

    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _failing_spot_em)
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons_3)
    monkeypatch.setattr("akshare.stock_zh_a_daily", _fake_daily)

    cache_dir = str(tmp_path)
    assert market_cap_symbols(2, cache_dir=cache_dir) == ["600000", "000001"]
    assert "falling back" in capsys.readouterr().out
    assert calls["n"] == 3

    # Second call: same day → served from cache, no further Sina requests.
    monkeypatch.setattr("akshare.stock_zh_a_daily",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError()))
    assert market_cap_symbols(2, cache_dir=cache_dir) == ["600000", "000001"]
    assert calls["n"] == 3


# ── pool_label (pure) ──────────────────────────────────────────────────

def test_pool_label():
    assert pool_label("csi300") == "沪深300成分股（全部）"
    assert pool_label("csi300:30") == "沪深300成分股（前 30）"
    assert pool_label("industry:白酒") == "行业(证监会) · 白酒"
    assert pool_label("market_cap:30") == "总市值最大（前 30）"
    assert pool_label("market_cap_small:30") == "总市值最小（前 30）"
    assert pool_label(["600519", "000001"]) == "自选列表（2 只）"
    assert pool_label("600519,000001") == "自选列表（2 只）"
    assert pool_label("600519") == "单只 · 600519"


def test_pool_label_valuation():
    assert pool_label("valuation:pe_low:30") == "估值 · PE(静) 最低（前 30）"
    assert pool_label("valuation:pe_ttm:20") == "估值 · PE(TTM) 最低（前 20）"
    assert pool_label("valuation:pb_high:10") == "估值 · PB 最高（前 10）"
    assert pool_label("valuation:dy:15") == "估值 · 股息率 最高（前 15）"


# ── symbol_names (mocked) ──────────────────────────────────────────────

def test_symbol_names(monkeypatch, tmp_path):
    def _fake_spot():
        return pd.DataFrame({
            "代码": ["sh600000", "sz000001", "bj920000"],
            "名称": ["浦发银行", "平安银行", "安徽凤凰"],
            "最新价": [10.0, 11.0, 12.0],
        })

    monkeypatch.setattr("akshare.stock_zh_a_spot", _fake_spot)
    names = symbol_names(["600000", "000001", "920000", "999999"],
                         universe_cache_dir=str(tmp_path),
                         valuation_cache_dir=str(tmp_path))
    assert names == {"600000": "浦发银行", "000001": "平安银行",
                     "920000": "安徽凤凰", "999999": "?"}


def test_symbol_names_graceful_failure(monkeypatch, tmp_path):
    """A failed lookup degrades to '?' rather than raising."""
    def _failing_spot():
        raise ConnectionError("boom")

    monkeypatch.setattr("akshare.stock_zh_a_spot", _failing_spot)
    names = symbol_names(["600000"],
                         universe_cache_dir=str(tmp_path),
                         valuation_cache_dir=str(tmp_path))
    assert names == {"600000": "?"}


def test_symbol_names_cache_first(monkeypatch, tmp_path):
    """Cached names (stock_basic + valuation) satisfy lookups with no
    network; only the missing codes hit the (mocked-failing) Sina spot.
    Exchange suffixes on the input are normalized to the 6-digit cache key."""
    sb = pl.DataFrame({
        "code": ["600000", "000001"],
        "name": ["浦发银行", "平安银行"],
    })
    sb.write_parquet(tmp_path / "stock_basic.parquet")
    val = pl.DataFrame({
        "code": ["600519"],
        "name": ["贵州茅台"],
    })
    val.write_parquet(tmp_path / "sw_valuation.parquet")

    def _failing_spot():
        raise ConnectionError("boom")

    monkeypatch.setattr("akshare.stock_zh_a_spot", _failing_spot)
    names = symbol_names(["600000.SH", "000001.SZ", "600519.SH", "999999"],
                         universe_cache_dir=str(tmp_path),
                         valuation_cache_dir=str(tmp_path))
    assert names == {"600000.SH": "浦发银行", "000001.SZ": "平安银行",
                     "600519.SH": "贵州茅台", "999999": "?"}


# ── Valuation stock pool (legulegu cache) ──────────────────────────────

def _write_valuation_cache(tmp_path, codes, pe=None, pe_ttm=None, pb=None,
                           dy=None, crawl_date=None):
    import os
    cache = tmp_path / "sw_valuation.parquet"
    n = len(codes)
    df = pl.DataFrame({
        "code": codes,
        "pe": pe if pe is not None else [20.0 - i for i in range(n)],
        "pe_ttm": pe_ttm if pe_ttm is not None else [None] * n,
        "pb": pb if pb is not None else [3.0 - 0.1 * i for i in range(n)],
        "dy": dy if dy is not None else [0.01 * i for i in range(n)],
    })
    if crawl_date is None:
        crawl_date = int(datetime.date.today().strftime("%Y%m%d"))
    df = df.with_columns(pl.lit(crawl_date).alias("crawl_date"))
    os.makedirs(tmp_path, exist_ok=True)
    df.write_parquet(cache)
    return tmp_path


def test_parse_valuation_spec():
    assert _parse_valuation_spec("pe_low:30") == ("pe", "low", 30)
    assert _parse_valuation_spec("pe_ttm:10") == ("pe_ttm", "low", 10)
    assert _parse_valuation_spec("pe_ttm_high:5") == ("pe_ttm", "high", 5)
    assert _parse_valuation_spec("pb_high:5") == ("pb", "high", 5)
    assert _parse_valuation_spec("dy:15") == ("dy", "high", 15)
    with pytest.raises(ValueError):
        _parse_valuation_spec("roe_low:5")
    with pytest.raises(ValueError):
        _parse_valuation_spec("pe_mid:5")
    with pytest.raises(ValueError):
        _parse_valuation_spec("pe_low:5:extra")


def test_valuation_symbols_pe_low(tmp_path):
    codes = ["600000", "000001", "300001", "688001", "600519"]
    cache = _write_valuation_cache(tmp_path, codes,
                                   pe=[20.0, 5.0, 30.0, 10.0, 15.0])
    got = valuation_symbols("pe_low:3", cache_dir=str(cache))
    # 5 < 10 < 15
    assert got == ["000001", "688001", "600519"]


def test_valuation_symbols_pb_high(tmp_path):
    codes = ["600000", "000001", "300001", "688001"]
    cache = _write_valuation_cache(tmp_path, codes,
                                   pb=[1.0, 4.0, 2.0, 3.0])
    got = valuation_symbols("pb_high:2", cache_dir=str(cache))
    assert got == ["000001", "688001"]


def test_valuation_symbols_dy_defaults_high(tmp_path):
    codes = ["600000", "000001", "300001"]
    cache = _write_valuation_cache(tmp_path, codes,
                                   dy=[0.03, 0.01, 0.05])
    got = valuation_symbols("dy:2", cache_dir=str(cache))
    assert got == ["300001", "600000"]


def test_valuation_symbols_excludes_nonpositive_and_null(tmp_path):
    """Loss-making (PE ≤ 0) and missing metrics are dropped from ranking."""
    codes = ["600000", "000001", "300001", "688001"]
    cache = _write_valuation_cache(tmp_path, codes,
                                   pe=[-5.0, 5.0, None, 15.0])
    got = valuation_symbols("pe_low:10", cache_dir=str(cache))
    assert got == ["000001", "688001"]
    assert "600000" not in got and "300001" not in got


def test_valuation_symbols_high_excludes_nan(tmp_path):
    """NaN rows must not surface at the top of a 'high' ranking (polars
    treats NaN > 0 as True)."""
    import math
    codes = ["600000", "000001", "300001"]
    cache = _write_valuation_cache(tmp_path, codes,
                                   pe=[float("nan"), 5.0, 15.0])
    got = valuation_symbols("pe_high:2", cache_dir=str(cache))
    assert got == ["300001", "000001"]
    assert "600000" not in got


def test_valuation_symbols_uses_cached_today(tmp_path, capsys):
    """A same-day cache is reused without a network crawl."""
    cache = _write_valuation_cache(tmp_path, ["600000", "000001"])
    got = valuation_symbols("pe_low:2", cache_dir=str(cache))
    assert len(got) == 2
    assert "Using cached valuation table" in capsys.readouterr().out


def test_valuation_symbols_stale_cache_fallback(monkeypatch, tmp_path, capsys):
    """A failed crawl falls back to the stale cache."""
    cache = _write_valuation_cache(tmp_path, ["600000", "000001", "300001"],
                                   pe=[1.0, 2.0, 3.0], crawl_date=20200101)

    def _failing_crawl(cache_dir=None):
        raise ConnectionError("legulegu down")

    monkeypatch.setattr("alftrade.datasource._crawl_sw_valuation_table",
                        _failing_crawl)
    monkeypatch.setattr(
        "alftrade.datasource.ts_valuation_table",
        lambda cache_dir: (_ for _ in ()).throw(ConnectionError("no token")))
    got = valuation_symbols("pe_low:2", cache_dir=str(cache))
    assert got == ["600000", "000001"]
    assert "reusing stale cache" in capsys.readouterr().out


def test_resolve_symbols_valuation(monkeypatch):
    def _fake_valuation(spec, cache_dir=None, refresh=False):
        assert spec == "pe_low:2"
        return ["600519", "000858"]

    monkeypatch.setattr("alftrade.datasource.valuation_symbols", _fake_valuation)
    assert resolve_symbols("valuation:pe_low:2") == ["600519", "000858"]


def test_backfill_valuation_names(monkeypatch, tmp_path, capsys):
    """An all-null name column is backfilled from Sina (symbol_names)."""
    import os
    from alftrade.datasource import _backfill_valuation_names
    cache_dir = str(tmp_path)
    today = int(datetime.date.today().strftime("%Y%m%d"))
    df = pl.DataFrame({
        "code": ["600519", "000858", "000001"],
        "name": [None, None, None],
        "pe": [20.0, 30.0, 5.0],
        "crawl_date": [today] * 3,
    })
    df.write_parquet(os.path.join(cache_dir, "sw_valuation.parquet"))
    monkeypatch.setattr("alftrade.datasource.symbol_names",
                        lambda codes: {"600519": "贵州茅台",
                                       "000858": "五粮液",
                                       "000001": "?"})  # Sina misses one
    assert _backfill_valuation_names(cache_dir) is True
    t = pl.read_parquet(os.path.join(cache_dir, "sw_valuation.parquet"))
    assert t["name"].to_list() == ["贵州茅台", "五粮液", None]
    assert "已从新浪快照补齐" in capsys.readouterr().out
    # names are no longer all-null → second call is a no-op
    assert _backfill_valuation_names(cache_dir) is False


def test_backfill_valuation_names_all_miss(monkeypatch, tmp_path):
    """No-op when the Sina source returns nothing usable."""
    import os
    from alftrade.datasource import _backfill_valuation_names
    cache_dir = str(tmp_path)
    today = int(datetime.date.today().strftime("%Y%m%d"))
    df = pl.DataFrame({
        "code": ["600519"], "name": [None], "pe": [20.0],
        "crawl_date": [today],
    })
    df.write_parquet(os.path.join(cache_dir, "sw_valuation.parquet"))
    monkeypatch.setattr("alftrade.datasource.symbol_names",
                        lambda codes: {"600519": "?"})
    assert _backfill_valuation_names(cache_dir) is False


# ── legulegu page parser (no network) ──────────────────────────────────

_SYNTHETIC_CONS_PAGE = """
<table>
  <thead><tr>
    <th>序号</th><th>股票代码</th><th>股票简称</th><th>纳入时间</th>
    <th>申万3级</th><th>细分概念</th>
    <th>价格  { "@context": "https://schema.org" }</th>
    <th>市盈率  { "@context": "https://schema.org" }</th>
    <th>市盈率ttm  { "@context": "https://schema.org" }</th>
    <th>市净率  { "@context": "https://schema.org" }</th>
    <th>股息率  { "@context": "https://schema.org" }</th>
    <th>市值（亿元）  { "@context": "https://schema.org" }</th>
  </tr></thead>
  <tbody>
    <tr><td>1</td><td>603589.SH</td><td>口子窖</td><td>2015-07-06</td>
        <td>白酒Ⅲ</td><td>—</td>
        <td>19.13</td><td>16.99</td><td>35.41</td><td>1.09</td>
        <td>2.61%</td><td>114.42</td></tr>
    <tr><td>2</td><td>000860.SZ</td><td>顺鑫农业</td><td>2000-01-04</td>
        <td>白酒Ⅲ</td><td>—</td>
        <td>10.59</td><td>—</td><td>—</td><td>1.16</td>
        <td>—</td><td>78.55</td></tr>
  </tbody>
</table>
"""


def test_parse_sw_cons_page():
    from alftrade.datasource import _parse_sw_cons_page
    df = _parse_sw_cons_page(_SYNTHETIC_CONS_PAGE)
    assert df["code"].to_list() == ["603589", "000860"]
    assert df["name"].to_list() == ["口子窖", "顺鑫农业"]
    assert df["pe"][0] == 16.99 and df["pe"].is_nan()[1]
    assert df["pe_ttm"][0] == 35.41 and df["pe_ttm"].is_nan()[1]
    assert df["pb"].to_list() == [1.09, 1.16]
    assert df["dy"][0] == 2.61 and df["dy"].is_nan()[1]
    assert df["cap"].to_list() == [114.42, 78.55]
