"""Tests for store.py: parquet-by-date write/read round-trip and lazy panel."""

import os
import polars as pl
import pytest

from alftrade import (
    generate_simulated_market,
    bars_to_frame,
    compute_factor_panel,
    factor_panel_to_dict,
    write_market_parquet,
    load_market_parquet,
    load_factor_panel,
    load_market_with_factors,
    ParquetPanel,
)


@pytest.fixture(scope="module")
def small_market():
    return generate_simulated_market(n_stocks=6, n_days=60, seed=5)


@pytest.fixture(scope="module")
def parquet_root(small_market, tmp_path_factory):
    bars, universe, market_returns = small_market
    root = str(tmp_path_factory.mktemp("market"))
    write_market_parquet(bars, universe, market_returns, root)
    return root


# ── Writing / layout ───────────────────────────────────────────────────

def test_writes_one_parquet_per_date(parquet_root, small_market):
    bars, _, _ = small_market
    marketdata_dir = os.path.join(parquet_root, "marketdata")
    files = sorted(os.listdir(marketdata_dir))
    assert len(files) == len(bars)
    assert files[0] == f"{sorted(bars)[0]}.parquet"


def test_writes_index_parquet(parquet_root, small_market):
    _, _, market_returns = small_market
    idx = pl.read_parquet(os.path.join(parquet_root, "index.parquet"))
    assert idx.columns == ["date", "market_return"]
    assert len(idx) == len(market_returns)


# ── ParquetPanel ───────────────────────────────────────────────────────

def test_panel_getitem_returns_daily_bar_dict(parquet_root):
    panel, universe, _ = load_market_parquet(parquet_root)
    d = panel.dates[0]
    bars = panel[d]
    assert isinstance(bars, dict)
    from alftrade import DailyBar
    assert all(isinstance(b, DailyBar) for b in bars.values())
    assert set(bars.keys()) == set(universe.symbols)


def test_panel_mapping_behavior(parquet_root):
    panel, _, _ = load_market_parquet(parquet_root)
    d = panel.dates[0]
    assert d in panel
    assert len(panel) == len(panel.dates)
    assert sorted(panel) == panel.dates
    # .get() from Mapping
    assert panel.get(d) is not None
    assert panel.get(-999, {}) == {}
    # Missing key raises KeyError
    with pytest.raises(KeyError):
        _ = panel[-999]


def test_panel_roundtrip_fields(parquet_root, small_market):
    bars, _, _ = small_market
    panel, _, _ = load_market_parquet(parquet_root)
    d = sorted(bars)[-1]
    original = bars[d]
    loaded = panel[d]
    for sym in original:
        o, l = original[sym], loaded[sym]
        assert l.open == o.open
        assert l.high == o.high
        assert l.low == o.low
        assert l.close == o.close
        assert l.volume == o.volume
        assert l.amount == o.amount
        assert l.prev_close == o.prev_close
        assert l.limit_up_price == o.limit_up_price
        assert l.limit_down_price == o.limit_down_price
        assert l.is_suspended == o.is_suspended
        assert l.total_shares == o.total_shares
        assert l.market_return == o.market_return  # injected from index.parquet


def test_panel_lru_caches(parquet_root):
    panel, _, _ = load_market_parquet(parquet_root)
    d = panel.dates[0]
    first = panel[d]
    second = panel[d]
    assert first is second  # cached → same object


# ── factor computation from parquet ────────────────────────────────────

def test_load_factor_panel_matches_in_memory(parquet_root, small_market):
    bars, _, _ = small_market
    # In-memory path
    factors_inmem = factor_panel_to_dict(compute_factor_panel(bars_to_frame(bars)))
    # Parquet path
    factors_disk = load_factor_panel(parquet_root)

    assert set(factors_inmem.keys()) == set(factors_disk.keys())
    d = sorted(bars)[-1]
    assert set(factors_inmem[d].keys()) == set(factors_disk[d].keys())
    for sym in factors_inmem[d]:
        for name in factors_inmem[d][sym]:
            assert factors_inmem[d][sym][name] == pytest.approx(
                factors_disk[d][sym][name], abs=1e-9)


def test_load_market_with_factors_signature(parquet_root, small_market):
    bars, universe, market_returns = small_market
    panel, factors, uni2, mkt2 = load_market_with_factors(parquet_root)
    assert set(uni2.symbols) == set(universe.symbols)
    assert set(mkt2.keys()) == set(market_returns.keys())
    assert len(factors) == len(bars)
    assert set(factors.keys()) == set(bars.keys())
