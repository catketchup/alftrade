"""Tests for stock-pool helpers: csi300_symbols, industry_symbols, resolve_symbols."""

import pandas as pd
import pytest

from alftrade import (
    csi300_symbols, industry_symbols, market_cap_symbols, resolve_symbols,
    pool_label, symbol_names,
)


def _fake_index_cons(symbol="000300"):
    return pd.DataFrame({
        "品种代码": [f"{600000 + i}" for i in range(10)],
        "品种名称": [f"股票{i}" for i in range(10)],
        "纳入日期": ["2026-06-15"] * 10,
    })


def _fake_industry_cons(symbol="白酒"):
    return pd.DataFrame({
        "代码": ["600519", "000858", "000568"],
        "名称": ["贵州茅台", "五粮液", "泸州老窖"],
    })


def _fake_spot_em():
    return pd.DataFrame({
        "代码": ["600000", "000001", "300001", "688001"],
        "名称": ["浦发银行", "平安银行", "小盘A", "小盘B"],
        "总市值": [3000.0, 2000.0, 100.0, 50.0],
        "流通市值": [2500.0, 1800.0, 90.0, 40.0],
    })


def test_csi300_symbols_all(monkeypatch):
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons)
    codes = csi300_symbols()
    assert codes == [f"{600000 + i}" for i in range(10)]


def test_csi300_symbols_slice(monkeypatch):
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons)
    codes = csi300_symbols(3)
    assert codes == ["600000", "600001", "600002"]


def test_industry_symbols(monkeypatch):
    monkeypatch.setattr("akshare.stock_board_industry_cons_em", _fake_industry_cons)
    codes = industry_symbols("白酒")
    assert codes == ["600519", "000858", "000568"]


def test_resolve_symbols_list_passthrough():
    assert resolve_symbols(["600519", "000001"]) == ["600519", "000001"]
    assert resolve_symbols(("600519",)) == ["600519"]


def test_resolve_symbols_csi300(monkeypatch):
    monkeypatch.setattr("akshare.index_stock_cons", _fake_index_cons)
    assert resolve_symbols("csi300") == [f"{600000 + i}" for i in range(10)]
    assert resolve_symbols("csi300:2") == ["600000", "600001"]


def test_resolve_symbols_industry(monkeypatch):
    monkeypatch.setattr("akshare.stock_board_industry_cons_em", _fake_industry_cons)
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


def test_resolve_symbols_market_cap(monkeypatch):
    monkeypatch.setattr("akshare.stock_zh_a_spot_em", _fake_spot_em)
    assert resolve_symbols("market_cap:2") == ["600000", "000001"]
    assert resolve_symbols("market_cap_small:2") == ["688001", "300001"]


def test_market_cap_symbols_sina_fallback(monkeypatch, capsys):
    """When Eastmoney is down, fall back to Sina per-stock market cap over CSI300."""
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

    assert market_cap_symbols(2) == ["600000", "000001"]
    assert "falling back" in capsys.readouterr().out


# ── pool_label (pure) ──────────────────────────────────────────────────

def test_pool_label():
    assert pool_label("csi300") == "沪深300成分股（全部）"
    assert pool_label("csi300:30") == "沪深300成分股（前 30）"
    assert pool_label("industry:白酒") == "行业 · 白酒"
    assert pool_label("market_cap:30") == "总市值最大（前 30）"
    assert pool_label("market_cap_small:30") == "总市值最小（前 30）"
    assert pool_label(["600519", "000001"]) == "自选列表（2 只）"
    assert pool_label("600519,000001") == "自选列表（2 只）"
    assert pool_label("600519") == "单只 · 600519"


# ── symbol_names (mocked) ──────────────────────────────────────────────

def test_symbol_names(monkeypatch):
    def _fake_spot():
        return pd.DataFrame({
            "代码": ["sh600000", "sz000001", "bj920000"],
            "名称": ["浦发银行", "平安银行", "安徽凤凰"],
            "最新价": [10.0, 11.0, 12.0],
        })

    monkeypatch.setattr("akshare.stock_zh_a_spot", _fake_spot)
    names = symbol_names(["600000", "000001", "920000", "999999"])
    assert names == {"600000": "浦发银行", "000001": "平安银行",
                     "920000": "安徽凤凰", "999999": "?"}


def test_symbol_names_graceful_failure(monkeypatch):
    """A failed lookup degrades to '?' rather than raising."""
    def _failing_spot():
        raise ConnectionError("boom")

    monkeypatch.setattr("akshare.stock_zh_a_spot", _failing_spot)
    assert symbol_names(["600000"]) == {"600000": "?"}
