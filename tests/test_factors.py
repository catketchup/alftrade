"""Tests for factors.py: polars-vectorized compute_factor_panel and
factor_panel_to_dict.

Each test builds a small long-format DataFrame (one row per symbol, date),
runs ``compute_factor_panel``, and checks factor values against hand-computed
expectations.
"""

import numpy as np
import polars as pl
import pytest

from alftrade.factors import FACTOR_NAMES, compute_factor_panel, factor_panel_to_dict


# ── Helpers ────────────────────────────────────────────────────────────

def _df_single(closes, market_returns=None, symbol="A", date_base=1):
    """Build a one-symbol long table from a list of closes."""
    rows = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        mkt = market_returns[i] if market_returns is not None else 0.0
        rows.append({
            "date": date_base + i, "symbol": symbol,
            "open": prev, "high": c * 1.001, "low": c * 0.999, "close": c,
            "volume": 1000.0, "amount": c * 1000.0,
            "prev_close": prev, "market_return": mkt,
        })
    return pl.DataFrame(rows)


def _last(df, name):
    """Return the factor value on the last row for the given column."""
    return df[name][-1]


# ── FACTOR_NAMES ───────────────────────────────────────────────────────

def test_factor_names_canonical():
    assert FACTOR_NAMES == [
        "daily_return", "overnight_return", "intraday_return",
        "momentum_5d", "momentum_20d", "ma_dev_20", "volatility_20d",
        "volume_ratio", "rsi_14", "atr_pct_14", "market_beta",
    ]


def test_factor_panel_columns_and_no_nan():
    """The factor panel has [date, symbol] + 11 factors and no NaN."""
    df = _df_single([100.0 + i for i in range(30)])
    out = compute_factor_panel(df)
    assert out.columns[:2] == ["date", "symbol"]
    assert out.columns[2:] == FACTOR_NAMES
    assert out.null_count().sum_horizontal()[0] == 0  # no nulls


# ── Price-based factors ────────────────────────────────────────────────

def test_daily_return():
    df = _df_single([100.0, 105.0])
    out = compute_factor_panel(df)
    assert _last(out, "daily_return") == pytest.approx(0.05)


def test_overnight_and_intraday_return():
    # Manually set open to distinguish overnight vs intraday.
    rows = [
        {"date": 1, "symbol": "A", "open": 100.0, "high": 105.0, "low": 100.0,
         "close": 105.0, "volume": 1000.0, "amount": 105000.0,
         "prev_close": 100.0, "market_return": 0.0},
        {"date": 2, "symbol": "A", "open": 102.0, "high": 106.0, "low": 102.0,
         "close": 105.0, "volume": 1000.0, "amount": 105000.0,
         "prev_close": 105.0, "market_return": 0.0},
    ]
    out = compute_factor_panel(pl.DataFrame(rows))
    row = out.row(-1, named=True)
    assert row["overnight_return"] == pytest.approx((102.0 - 105.0) / 105.0)
    assert row["intraday_return"] == pytest.approx((105.0 - 102.0) / 102.0)


# ── Momentum ───────────────────────────────────────────────────────────

def test_momentum_5d():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
    out = compute_factor_panel(_df_single(closes))
    assert _last(out, "momentum_5d") == pytest.approx((110.0 - 100.0) / 100.0)


def test_momentum_20d():
    closes = [100.0 + i * 0.5 for i in range(21)]
    out = compute_factor_panel(_df_single(closes))
    assert _last(out, "momentum_20d") == pytest.approx((110.0 - 100.0) / 100.0)


# ── ma_dev_20 ──────────────────────────────────────────────────────────

def test_ma_dev_20():
    closes = [100.0] * 20 + [110.0]
    out = compute_factor_panel(_df_single(closes))
    # rolling_mean(20) at row 21 = mean of rows 2..21 = (19*100 + 110)/20
    sma = (19 * 100.0 + 110.0) / 20
    assert _last(out, "ma_dev_20") == pytest.approx((110.0 - sma) / sma)


# ── Volatility ─────────────────────────────────────────────────────────

def test_volatility_20d_nonneg_and_positive():
    # Alternating returns → nonzero volatility.
    closes = [100.0]
    prev = 100.0
    for i in range(20):
        prev = prev * (1.0 + (0.01 if i % 2 == 0 else -0.005))
        closes.append(prev)
    out = compute_factor_panel(_df_single(closes))
    assert _last(out, "volatility_20d") > 0


# ── Volume ratio ───────────────────────────────────────────────────────

def test_volume_ratio():
    rows = []
    prev = 100.0
    for i in range(20):
        vol = 1000.0 if i < 19 else 2000.0
        rows.append({"date": i + 1, "symbol": "A", "open": prev, "high": 101.0,
                     "low": 99.0, "close": 100.0, "volume": vol,
                     "amount": vol * 100.0, "prev_close": prev,
                     "market_return": 0.0})
    out = compute_factor_panel(pl.DataFrame(rows))
    # rolling_mean(20) of volume at last row = (19*1000 + 2000)/20 = 1050
    assert _last(out, "volume_ratio") == pytest.approx(2000.0 / 1050.0)


# ── RSI ────────────────────────────────────────────────────────────────

def test_rsi_range():
    closes = [100.0]
    prev = 100.0
    for i in range(30):
        prev = prev * (1.0 + (0.001 if i % 2 else -0.001))
        closes.append(prev)
    out = compute_factor_panel(_df_single(closes))
    assert -1.0 <= _last(out, "rsi_14") <= 1.0


def test_rsi_pure_uptrend_positive():
    closes = [100.0 * (1.01 ** i) for i in range(30)]
    out = compute_factor_panel(_df_single(closes))
    # All gains → RSI raw ≈ 100 → rescaled ≈ +1.0.
    assert _last(out, "rsi_14") > 0.9


# ── ATR ────────────────────────────────────────────────────────────────

def test_atr_pct_14_positive():
    rows = []
    prev = 100.0
    for i in range(20):
        rows.append({"date": i + 1, "symbol": "A", "open": 100.0, "high": 101.0,
                     "low": 99.0, "close": 100.0, "volume": 1000.0,
                     "amount": 100000.0, "prev_close": prev,
                     "market_return": 0.0})
        prev = 100.0
    out = compute_factor_panel(pl.DataFrame(rows))
    assert _last(out, "atr_pct_14") > 0


# ── Market beta ────────────────────────────────────────────────────────

def test_market_beta_recovers_loading():
    """daily_return = 1.2 × market_return + noise → beta ≈ 1.2."""
    rng = np.random.default_rng(42)
    market = rng.normal(0.0, 0.01, 60)
    returns = 1.2 * market + rng.normal(0.0, 0.002, 60)
    df = _df_single([100.0], market_returns=[0.0])
    # Rebuild with explicit returns.
    rows = []
    close = 100.0
    for i in range(60):
        prev = close
        close = prev * (1.0 + returns[i])
        rows.append({"date": i + 1, "symbol": "A", "open": prev,
                     "high": close * 1.001, "low": close * 0.999, "close": close,
                     "volume": 1000.0, "amount": close * 1000.0,
                     "prev_close": prev, "market_return": market[i]})
    out = compute_factor_panel(pl.DataFrame(rows))
    assert 1.0 <= _last(out, "market_beta") <= 1.4


def test_market_beta_warmup_neutral():
    """With no market variation, beta falls back to 1.0."""
    df = _df_single([100.0 + i * 0.1 for i in range(30)])  # market_return all 0
    out = compute_factor_panel(df)
    # market_return has zero variance → beta = 1.0 (fallback).
    assert _last(out, "market_beta") == 1.0


# ── factor_panel_to_dict ───────────────────────────────────────────────

def test_factor_panel_to_dict_structure():
    df = pl.concat([
        _df_single([100.0 + i for i in range(5)], symbol="A"),
        _df_single([50.0 + i for i in range(5)], symbol="B"),
    ])
    out = compute_factor_panel(df)
    d = factor_panel_to_dict(out)
    assert set(d.keys()) == {1, 2, 3, 4, 5}
    assert set(d[5].keys()) == {"A", "B"}
    assert set(d[5]["A"].keys()) == set(FACTOR_NAMES)
    assert isinstance(d[5]["A"]["daily_return"], float)
