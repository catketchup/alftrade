"""
schema.py - The canonical data contract shared by every data source and the
factor layer.

Every data source (the simulated generator, ``ingest_parquet``, akshare,
baostock, or any future vendor) must emit a long-format table with the exact
columns defined here.  The factor layer consumes that table and is otherwise
agnostic to *where* the rows came from — so swapping the simulated generator
for real data leaves the factor computation and everything downstream
unchanged.

Two column sets matter:

  * ``BAR_COLUMNS`` — the full daily-bar table (14 columns). This is what
    ``data.bars_to_frame`` produces and ``data.frame_to_bars`` consumes; it
    carries every field of ``DailyBar`` plus the index return.

  * ``FACTOR_INPUT_COLUMNS`` — the 9-column subset that
    ``factors.compute_factor_panel`` actually reads. A source only has to
    satisfy these to be usable for factor computation; the remaining bar
    fields are only needed by the portfolio / risk layers.
"""

# Full long-table columns for one daily bar (mirrors DailyBar + market_return).
BAR_COLUMNS: list[str] = [
    "date", "symbol", "open", "high", "low", "close", "volume", "amount",
    "prev_close", "limit_up_price", "limit_down_price", "is_suspended",
    "total_shares", "market_return",
]

# Columns stored per date in marketdata/*.parquet (market_return lives in
# index.parquet because it is shared across all symbols on a date).
MARKETDATA_COLUMNS: list[str] = [
    "date", "symbol", "open", "high", "low", "close", "volume", "amount",
    "prev_close", "limit_up_price", "limit_down_price", "is_suspended",
    "total_shares",
]

# Columns required by factors.compute_factor_panel.
FACTOR_INPUT_COLUMNS: list[str] = [
    "date", "symbol", "open", "high", "low", "close", "volume",
    "prev_close", "market_return",
]


def missing_columns(df_columns, required: list[str]) -> list[str]:
    """Return the ``required`` columns absent from ``df_columns``."""
    have = set(df_columns)
    return [c for c in required if c not in have]


def validate_factor_inputs(df_columns) -> None:
    """Raise ``ValueError`` if a frame lacks any column the factor layer needs."""
    missing = missing_columns(df_columns, FACTOR_INPUT_COLUMNS)
    if missing:
        raise ValueError(
            f"Factor computation requires columns {FACTOR_INPUT_COLUMNS}; "
            f"missing {missing}. Ensure the data source emits the canonical "
            f"schema (see alftrade.schema)."
        )


def validate_bar_frame(df_columns) -> None:
    """Raise ``ValueError`` if a frame lacks any column of ``BAR_COLUMNS``."""
    missing = missing_columns(df_columns, BAR_COLUMNS)
    if missing:
        raise ValueError(
            f"Expected the canonical bar schema {BAR_COLUMNS}; missing {missing}."
        )
