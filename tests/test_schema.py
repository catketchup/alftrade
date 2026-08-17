"""Tests for schema.py and the source-agnostic data → factor contract.

Proves that the factor layer and everything downstream are independent of the
data source: the same pipeline runs identically on simulated data and on
ingested real data, because both emit the canonical ``schema`` columns.
"""

import os
import polars as pl
import pytest

from alftrade import (
    simulate_to_frame, ingest_parquet,
    compute_factor_panel, factor_panel_to_dict,
    TradingEngine, FACTOR_NAMES,
)
from alftrade.schema import (
    BAR_COLUMNS, MARKETDATA_COLUMNS, FACTOR_INPUT_COLUMNS,
    validate_factor_inputs, validate_bar_frame,
)


# ── Schema constants ───────────────────────────────────────────────────

def test_schema_column_sets():
    """FACTOR_INPUT_COLUMNS ⊆ BAR_COLUMNS; MARKETDATA = BAR minus market_return."""
    assert set(FACTOR_INPUT_COLUMNS) <= set(BAR_COLUMNS)
    assert set(MARKETDATA_COLUMNS) == set(BAR_COLUMNS) - {"market_return"}
    assert set(BAR_COLUMNS) == set(MARKETDATA_COLUMNS) | {"market_return"}


def test_validate_factor_inputs_missing_column():
    df = pl.DataFrame({"date": [1], "symbol": ["A"], "close": [10.0]})
    with pytest.raises(ValueError, match="market_return"):
        validate_factor_inputs(df.columns)


def test_compute_factor_panel_rejects_missing_column():
    df = pl.DataFrame({"date": [1], "symbol": ["A"], "close": [10.0]})
    with pytest.raises(ValueError, match="missing"):
        compute_factor_panel(df)


# ── Both sources emit the canonical schema ─────────────────────────────

def test_simulated_source_emits_canonical_schema():
    df = simulate_to_frame(n_stocks=8, n_days=60, seed=1)
    assert set(df.columns) == set(BAR_COLUMNS)


def test_real_source_emits_canonical_schema(tmp_path):
    _write_intraday_parquet(str(tmp_path))
    df = ingest_parquet(str(tmp_path), intraday=True)
    assert set(df.columns) == set(BAR_COLUMNS)


# ── Source-agnostic pipeline ───────────────────────────────────────────

def test_factor_pipeline_is_source_agnostic(tmp_path):
    """Simulated and ingested-real data produce structurally identical factors
    and both feed the engine unchanged."""
    sim_df = simulate_to_frame(n_stocks=8, n_days=60, seed=1)
    _write_intraday_parquet(str(tmp_path))
    real_df = ingest_parquet(str(tmp_path), intraday=True)

    for df in (sim_df, real_df):
        factors_df = compute_factor_panel(df)
        assert factors_df.columns == ["date", "symbol"] + FACTOR_NAMES

        factors = factor_panel_to_dict(factors_df)
        engine = TradingEngine(factors=factors)
        bars_by_date, _, _ = _frame_to_bars(df)

        # The engine builds a cross-section for every date from either source.
        for d in sorted(bars_by_date):
            cs = engine.build_cross_section(d, bars_by_date[d])
            assert cs.z_matrix.shape[1] == len(FACTOR_NAMES)


# ── Helpers ────────────────────────────────────────────────────────────

def _write_intraday_parquet(root):
    rows = []
    for date, sym, close in [
        ("2024-09-02", "000001.SZ", 100.0),
        ("2024-09-02", "600000.SH", 50.0),
        ("2024-09-03", "000001.SZ", 101.0),
        ("2024-09-03", "600000.SH", 49.0),
    ]:
        rows.append({"datetime": f"{date} 15:00:00", "stock_code": sym,
                     "open": close, "high": close, "low": close, "close": close,
                     "volume": 1000.0, "amount": close * 1000.0})
    df = pl.DataFrame(rows).with_columns(pl.col("datetime").str.to_datetime())
    for date, day in df.group_by(pl.col("datetime").dt.strftime("%Y%m%d")):
        day.write_parquet(os.path.join(root, f"{date}.parquet"))


def _frame_to_bars(df):
    from alftrade import frame_to_bars
    return frame_to_bars(df)
