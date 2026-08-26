"""
datasource.py - Unified, swappable data-source layer.

Provides a single entry point that returns the canonical long table
(``schema.BAR_COLUMNS``) regardless of the data source:

  * ``"simulated"`` — generated on demand and cached as parquet-by-date on
    disk.  The generation logic lives here (not in the demo), and re-runs are
    skipped with a notice when the cache already exists.
  * a directory path — real A-share parquet-by-date data, ingested at
    **daily-bar** granularity by default (``ingest_parquet``), matching the
    simulated data's granularity.

Switching sources is a single argument to ``load_panel``; the factor layer and
everything downstream are unchanged.
"""

import glob
import os
from typing import Optional

import polars as pl

from alftrade.data import generate_simulated_market, frame_to_bars
from alftrade.store import write_market_parquet, read_market_frame
from alftrade.ingest import ingest_parquet, ingest_akshare


# Default disk location for the cached simulated panel.
DEFAULT_SIMULATED_ROOT = "data/simulated"


def _retry(fn, attempts: int = 3, delay: float = 1.5):
    """Call ``fn`` with a few retries (akshare endpoints are rate-limited)."""
    import time

    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — surface the last network error
            last = e
            time.sleep(delay)
    raise last


def csi300_symbols(n: Optional[int] = None) -> list[str]:
    """Return CSI300 constituent codes (6-digit, no exchange suffix).

    Requires network (akshare). Returns the first ``n`` codes (ascending by
    code), or all 300 if ``n`` is None.

    Example:
        csi300_symbols(30)  # first 30 CSI300 members
    """
    import akshare as ak

    cons = _retry(lambda: ak.index_stock_cons(symbol="000300"), attempts=5, delay=2.0)
    codes = cons["品种代码"].astype(str).tolist()
    return codes[:n] if n is not None else codes


def industry_symbols(industry: str) -> list[str]:
    """Return the 6-digit codes of A-share stocks in an Eastmoney industry board.

    Requires network (akshare). ``industry`` is a board name such as "白酒",
    "半导体", "银行" (see ``ak.stock_board_industry_name_ths`` for the full list).

    Note: this endpoint (Eastmoney) can be rate-limited; a few retries are
    attempted before raising.
    """
    import akshare as ak

    cons = _retry(lambda: ak.stock_board_industry_cons_em(symbol=industry))
    return cons["代码"].astype(str).tolist()


def _sina_market_caps(codes: list[str]) -> dict[str, float]:
    """Total market cap (yuan) per code = latest close × outstanding_share.

    Uses the Sina daily source (``stock_zh_a_daily``), which is not subject to
    the Eastmoney ``push2`` rate-limit.  Per-stock, so this is slower than the
    Eastmoney spot; used only as a fallback.  Failed lookups map to NaN.
    """
    import akshare as ak
    import time
    import datetime

    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y%m%d")
    caps: dict[str, float] = {}
    for code in codes:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        try:
            raw = ak.stock_zh_a_daily(
                symbol=prefix + code, start_date=start, end_date=end, adjust="")
            last = raw.iloc[-1]
            caps[code] = float(last["close"] * last["outstanding_share"])
        except Exception:
            caps[code] = float("nan")
        time.sleep(0.05)
    return caps


def market_cap_symbols(
    n: int,
    ascending: bool = False,
    float_cap: bool = False,
) -> list[str]:
    """Return the top/bottom ``n`` A-share codes by market cap.

    Requires network (akshare). Uses 总市值 (total market cap) by default, or
    流通市值 (free-float cap) if ``float_cap=True``.

    ``ascending=False`` → the ``n`` largest (large-cap pool);
    ``ascending=True`` → the ``n`` smallest (small-cap pool).

    Two-tier strategy:

      * Primary: Eastmoney spot (``stock_zh_a_spot_em``) — one bulk call, has
        总市值/流通市值.
      * Fallback: when the Eastmoney ``push2`` endpoint is rate-limited, falls
        back to the Sina daily source over the CSI300 universe (large-cap pool
        ⊆ CSI300), computing 总市值 = close × outstanding_share per stock.  The
        fallback is slower (~0.5 s per stock) and only provides *total* cap
        (``float_cap`` is ignored), but it works when Eastmoney is down.
    """
    import math
    import akshare as ak

    col = "流通市值" if float_cap else "总市值"
    try:
        spot = _retry(lambda: ak.stock_zh_a_spot_em(), attempts=3, delay=1.5)
        df = spot.sort_values(col, ascending=ascending, na_position="last")
        return df["代码"].astype(str).head(n).tolist()
    except Exception:
        print("[data] Eastmoney spot unavailable — falling back to Sina daily "
              "(per-stock market cap over CSI300, ~30s–2min) …")
        codes = csi300_symbols()
        caps = _sina_market_caps(codes)
        valid = [(c, caps[c]) for c in codes if not math.isnan(caps[c])]
        valid.sort(key=lambda t: t[1], reverse=not ascending)
        return [c for c, _ in valid[:n]]


def resolve_symbols(spec) -> list[str]:
    """Resolve a stock-pool spec into a list of 6-digit codes.

    ``spec`` may be:

      * a list/tuple of codes (returned as-is, e.g. ``["600519", "000001"]``),
      * ``"csi300"`` or ``"csi300:30"`` — all / first N CSI300 constituents,
      * ``"industry:<name>"`` — Eastmoney industry-board constituents,
      * ``"market_cap:30"`` / ``"market_cap_small:30"`` — top/bottom N by
        market cap,
      * ``"600519,000001"`` — a comma-separated code string,
      * a single bare code (``"600519"``) — wrapped into a one-element list.

    This is the single entry point used by ``fetch_daily_to_parquet`` callers
    who want a configurable stock pool.
    """
    if isinstance(spec, (list, tuple)):
        return [str(s) for s in spec]
    if not isinstance(spec, str):
        raise ValueError(f"Unrecognized stock-pool spec: {spec!r}")
    if spec == "csi300" or spec.startswith("csi300:"):
        n = int(spec.split(":", 1)[1]) if ":" in spec else None
        return csi300_symbols(n)
    if spec.startswith("industry:"):
        return industry_symbols(spec.split(":", 1)[1])
    if spec.startswith("market_cap_small:"):
        return market_cap_symbols(int(spec.split(":", 1)[1]), ascending=True)
    if spec == "market_cap" or spec.startswith("market_cap:"):
        n = int(spec.split(":", 1)[1]) if ":" in spec else 50
        return market_cap_symbols(n)
    if "," in spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    return [spec]


def pool_label(spec) -> str:
    """Return a human-readable label for a stock-pool spec.

    Pure (no network) — mirrors the ``resolve_symbols`` dispatch so the demo can
    annotate a pool without re-deriving it.
    """
    if isinstance(spec, (list, tuple)):
        return f"自选列表（{len(spec)} 只）"
    if not isinstance(spec, str):
        return str(spec)
    if spec == "csi300":
        return "沪深300成分股（全部）"
    if spec.startswith("csi300:"):
        return f"沪深300成分股（前 {spec.split(':', 1)[1]}）"
    if spec.startswith("industry:"):
        return f"行业 · {spec.split(':', 1)[1]}"
    if spec.startswith("market_cap_small:"):
        return f"总市值最小（前 {spec.split(':', 1)[1]}）"
    if spec == "market_cap" or spec.startswith("market_cap:"):
        n = spec.split(":", 1)[1] if ":" in spec else "50"
        return f"总市值最大（前 {n}）"
    if "," in spec:
        return f"自选列表（{len(spec.split(','))} 只）"
    return f"单只 · {spec}"


def symbol_names(codes: list[str]) -> dict[str, str]:
    """Map 6-digit codes to stock names via the stable Sina spot source.

    Requires network (akshare, Sina spot). Returns ``{code: name}``; unknown
    codes, or a failed lookup (e.g. the source is rate-limited), map to ``"?"``
    — this function never raises, so it can't abort a pipeline just because
    name decoration is unavailable.
    """
    import akshare as ak

    try:
        spot = _retry(lambda: ak.stock_zh_a_spot(), attempts=2, delay=1.0)
    except Exception:
        return {c: "?" for c in codes}

    name_map = {
        code[-6:]: name
        for code, name in zip(spot["代码"].astype(str), spot["名称"].astype(str))
    }
    return {c: name_map.get(c, "?") for c in codes}


def _has_generated(root: str) -> bool:
    """True if ``root/marketdata`` already contains generated parquet files."""
    marketdata_dir = os.path.join(root, "marketdata")
    return os.path.isdir(marketdata_dir) and len(os.listdir(marketdata_dir)) > 0


def _is_cache(root: str) -> bool:
    """True if ``root`` is a parquet cache (``marketdata/`` + ``index.parquet``).

    Distinguishes the cache layout written by ``load_simulated`` /
    ``fetch_daily_to_parquet`` from a raw vendor directory (per-date parquet
    files at the top level with ``datetime`` / ``stock_code`` columns).
    """
    return (os.path.isfile(os.path.join(root, "index.parquet"))
            and os.path.isdir(os.path.join(root, "marketdata")))


def load_simulated(
    root: str = DEFAULT_SIMULATED_ROOT,
    n_stocks: int = 30,
    n_days: int = 250,
    seed: int = 7,
    force: bool = False,
) -> pl.DataFrame:
    """Generate (if needed) simulated data as parquet-by-date, and return the long table.

    First checks whether ``root/marketdata`` already has generated parquet:
      * present and ``force=False`` → reuse, printing a notice.
      * otherwise → generate + write parquet, printing a notice, then load.

    The generation code is isolated here — callers (and the demo) never touch
    ``generate_simulated_market`` / ``write_market_parquet`` directly.

    Args:
        root: Directory for the parquet cache (``marketdata/*.parquet`` +
            ``index.parquet``).
        n_stocks, n_days, seed: Passed to ``generate_simulated_market``.
        force: If True, regenerate even when a cache already exists.

    Returns:
        The canonical long table (``schema.BAR_COLUMNS``).
    """
    if _has_generated(root) and not force:
        print(f"[data] Using existing simulated data at '{root}' "
              f"({len(os.listdir(os.path.join(root, 'marketdata')))} files)")
    else:
        print(f"[data] Simulated data not found at '{root}' — generating "
              f"{n_stocks} stocks × {n_days} days ...")
        bars, universe, market_returns = generate_simulated_market(
            n_stocks=n_stocks, n_days=n_days, seed=seed)
        write_market_parquet(bars, universe, market_returns, root)
        print(f"[data] Wrote {n_days} daily parquet files to '{root}/marketdata'")

    return read_market_frame(root)


def load_real(
    source_dir: str,
    intraday: bool = False,
    limit_pct: float = 0.10,
) -> pl.DataFrame:
    """Ingest real parquet-by-date data into the canonical long table.

    Defaults to **daily-bar** granularity (``intraday=False``), matching the
    simulated data.  Pass ``intraday=True`` to aggregate 1-minute bars.

    Args:
        source_dir: Directory of ``*.parquet`` files (one per trading date).
        intraday: True if the source is intraday (1-minute) bars.
        limit_pct: Daily price-limit fraction for limit-up/down derivation.

    Returns:
        The canonical long table (``schema.BAR_COLUMNS``).
    """
    return ingest_parquet(source_dir, intraday=intraday, limit_pct=limit_pct)


def _cache_symbols(root: str) -> list[str]:
    """6-digit codes stored in the cache (from the newest marketdata parquet)."""
    files = sorted(glob.glob(os.path.join(root, "marketdata", "*.parquet")))
    if not files:
        return []
    df = pl.read_parquet(files[-1])
    return sorted(df["symbol"].str.slice(0, 6).unique().to_list())


def _cache_matches(root: str, symbols: list[str]) -> bool:
    """True if the cache was built from exactly the requested stock pool."""
    return _cache_symbols(root) == sorted(str(s) for s in symbols)


def fetch_daily_to_parquet(
    symbols: list[str],
    start_date: str,
    end_date: str,
    root: str,
    limit_pct: Optional[float] = None,
    force: bool = False,
) -> pl.DataFrame:
    """Fetch real daily bars (akshare) and cache them as parquet-by-date.

    Symmetric with ``load_simulated``: checks ``root/marketdata`` first and
    reuses it (with a notice); otherwise fetches via ``ingest_akshare``, writes
    parquet, and reads back.  After this, ``load_panel(root)`` returns the same
    data with no network access.

    The cache is keyed by the stock pool: if the cached symbols differ from the
    requested ``symbols``, it re-fetches (so changing ``STOCK_POOL`` actually
    takes effect).

    Args:
        symbols: A-share 6-digit codes without exchange suffix (e.g. "000001").
        start_date / end_date: "YYYYMMDD" range.
        root: Directory for the parquet cache.
        limit_pct: Daily price-limit fraction. If None (default), the
            board-specific limit is used (10% / 20% / 30% by code prefix).
        force: If True, re-fetch even when a cache already exists.

    Returns:
        The canonical long table (``schema.BAR_COLUMNS``).
    """
    reuse = _has_generated(root) and not force and _cache_matches(root, symbols)
    if reuse:
        print(f"[data] Using existing real daily data at '{root}' "
              f"({len(os.listdir(os.path.join(root, 'marketdata')))} files)")
    else:
        if _has_generated(root):
            print(f"[data] Cached pool differs from the requested {len(symbols)} "
                  f"stocks — re-fetching ...")
        else:
            print(f"[data] Fetching {len(symbols)} stocks × {start_date}..{end_date} "
                  f"via akshare (daily) ...")
        df = ingest_akshare(symbols, start_date, end_date, limit_pct=limit_pct)
        bars, universe, market_returns = frame_to_bars(df)
        write_market_parquet(bars, universe, market_returns, root)
        print(f"[data] Wrote {len(bars)} daily parquet files to '{root}/marketdata'")

    return read_market_frame(root)


def load_panel(source: str = "simulated", **kwargs) -> pl.DataFrame:
    """Unified data-source switch. Returns the canonical long table.

    Args:
        source: ``"simulated"`` (uses the cached/generated simulated panel), or
            a directory path to real parquet-by-date data.
        **kwargs: Forwarded to ``load_simulated`` (when ``"simulated"``, e.g.
            ``root``, ``n_stocks``, ``n_days``, ``seed``, ``force``) or to
            ``load_real`` (e.g. ``intraday``, ``limit_pct``).

    Returns:
        The canonical long table (``schema.BAR_COLUMNS``) for either source.

    Example:
        df = load_panel("simulated", root="data/simulated")   # simulated
        df = load_panel("/path/to/daily/real/data")           # raw vendor dir (daily)
        df = load_panel("/path/to/cached/root")               # existing parquet cache
    """
    if source == "simulated":
        return load_simulated(**kwargs)
    if not os.path.isdir(source):
        raise ValueError(
            f"Unknown data source '{source}': pass \"simulated\" or an "
            f"existing directory of parquet-by-date data."
        )
    # An existing cache (marketdata/ + index.parquet) is read directly;
    # otherwise the directory is treated as a raw vendor source to ingest.
    if _is_cache(source):
        return read_market_frame(source)
    return load_real(source, **kwargs)
