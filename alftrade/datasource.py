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
import re
from typing import Optional

import polars as pl

from alftrade.data import generate_simulated_market, frame_to_bars
from alftrade.store import write_market_parquet, read_market_frame
from alftrade.ingest import ingest_parquet, ingest_akshare
from alftrade.ts_source import ts_industry_symbols, ts_valuation_table


# Default disk location for the cached simulated panel.
DEFAULT_SIMULATED_ROOT = "data/simulated"

# Default disk location for universe-level caches (CSI300 / caps / names).
DEFAULT_UNIVERSE_CACHE_DIR = "data/universe"


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


def csi300_symbols(
    n: Optional[int] = None,
    cache_dir: str = DEFAULT_UNIVERSE_CACHE_DIR,
) -> list[str]:
    """Return CSI300 constituent codes, cached per day.

    Requires network (akshare, Sina index page).  Returns the first ``n``
    codes (ascending by code), or all 300 if ``n`` is None.  Results are
    cached to ``cache_dir/csi300.parquet``; a same-day cache hit needs no
    network, and a fetch failure reuses a stale cache when available.
    """
    import datetime
    import akshare as ak

    cache = os.path.join(cache_dir, "csi300.parquet")
    today = int(datetime.date.today().strftime("%Y%m%d"))
    cached: Optional[list[str]] = None
    if os.path.isfile(cache):
        try:
            df = pl.read_parquet(cache)
            if int(df["crawl_date"].max()) == today:
                codes = df["code"].to_list()
                return codes[:n] if n is not None else codes
            cached = df["code"].to_list()
        except Exception:  # noqa: BLE001 — treat a corrupt cache as absent
            pass
    try:
        cons = _retry(lambda: ak.index_stock_cons(symbol="000300"),
                      attempts=5, delay=2.0)
        codes = cons["品种代码"].astype(str).tolist()
        os.makedirs(cache_dir, exist_ok=True)
        pl.DataFrame({
            "code": codes,
            "crawl_date": [today] * len(codes),
        }).write_parquet(cache)
        return codes[:n] if n is not None else codes
    except Exception as e:
        if cached is not None:
            print(f"[data] 沪深300成分获取失败 ({type(e).__name__}: "
                  f"{str(e)[:60]}) — 复用陈旧缓存 …")
            return cached[:n] if n is not None else cached
        raise RuntimeError(
            f"无法获取沪深300成分 (新浪指数页不可用且无缓存): "
            f"{type(e).__name__}: {str(e)[:80]}") from e


def industry_symbols(industry: str) -> list[str]:
    """Return the 6-digit codes of A-share stocks matching an industry keyword.

    Built from the Tushare ``stock_basic`` table (证监会行业分类 ``industry``
    field plus a stock-name match), cached per day.  The 证监会 classification
    is coarse — all liquor stocks share "酒、饮料和精制茶制造业" — so
    ``industry_symbols("酒")`` returns the broad sector while a finer keyword
    like "白酒" only matches stock names containing those characters.  Tushare
    exposes no industry-constituent interface (THS/SW boards) at low point
    tiers.

    Requires ``TUSHARE_TOKEN`` in the environment.  Raises with a hint when
    the keyword matches nothing.
    """
    return ts_industry_symbols(industry)


# ── Per-stock valuation (legulegu 申万三级成分页) ───────────────────────
#
# Eastmoney's spot (总市值/市盈率/市净率) is rate-limited, so per-stock
# valuation comes from the legulegu 申万三级 constituent pages, which are
# server-rendered and carry per-stock 市盈率 / 市盈率ttm / 市净率 / 股息率 /
# 市值.  The 335 pages are crawled once per day and cached to parquet.

DEFAULT_VALUATION_CACHE_DIR = "data/valuation"

_VALUATION_METRICS = {"pe", "pe_ttm", "pb", "dy"}
_VALUATION_LABELS = {"pe": "PE(静)", "pe_ttm": "PE(TTM)", "pb": "PB", "dy": "股息率"}
_DEFAULT_DIRECTION = {"pe": "low", "pe_ttm": "low", "pb": "low", "dy": "high"}
# legulegu page column names → canonical cache columns.
_SW_VALUATION_RENAMES = {
    "市盈率": "pe",
    "市盈率ttm": "pe_ttm",
    "市净率": "pb",
    "股息率": "dy",
    "市值（亿元）": "cap",
}
_HEADER_METRIC = re.compile(r"^([^\s{]+)")


def _parse_sw_cons_page(text: str):
    """Parse a legulegu 申万三级成分页 into {code, name, valuation cols}.

    The page template annotates every valuation column header with embedded
    schema.org JSON, so the metric name is extracted with a regex and numeric
    cells are coerced after stripping "%"/"—" markers.
    """
    import io

    import pandas as pd

    raw = pd.read_html(io.StringIO(text))[0]
    code = raw.iloc[:, 1].astype(str).str.extract(r"(\d{6})")[0]
    df = pl.DataFrame({"code": code, "name": raw.iloc[:, 2].astype(str)})
    extra = {}
    for col in raw.columns[6:]:
        m = _HEADER_METRIC.match(str(col))
        name = m.group(1) if m else None
        if name in _SW_VALUATION_RENAMES:
            values = (
                raw[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace("—", "", regex=False)
                .str.replace("-", "", regex=False)
            )
            extra[_SW_VALUATION_RENAMES[name]] = pd.to_numeric(
                values, errors="coerce").to_numpy()
    return df.with_columns([pl.Series(k, v).alias(k) for k, v in extra.items()])


def _crawl_sw_valuation_table(cache_dir: str = DEFAULT_VALUATION_CACHE_DIR) -> pl.DataFrame:
    """Crawl all 申万三级 constituent pages (legulegu) into one valuation table.

    One request per third-level industry (~335 pages, all rows on one page);
    failed pages are skipped, but the crawl raises if more than half fail.
    The industry-code list is cached separately (``sw_industries.parquet``) so
    the crawl can still run when the overview page is down.
    """
    import datetime
    import time

    import akshare as ak
    import requests

    industry_cache = os.path.join(cache_dir, "sw_industries.parquet")
    codes: Optional[list[str]] = None
    try:
        info = _retry(lambda: ak.sw_index_third_info(), attempts=3, delay=2.0)
        codes = info["行业代码"].astype(str).tolist()
        os.makedirs(cache_dir, exist_ok=True)
        pl.DataFrame({"industry_code": codes}).write_parquet(industry_cache)
    except Exception:
        if os.path.isfile(industry_cache):
            print("[data] 申万行业列表页不可用 — 使用缓存的行业代码列表 …")
            codes = pl.read_parquet(industry_cache)["industry_code"].to_list()
    if not codes:
        raise RuntimeError(
            "无法获取申万三级行业代码列表 (legulegu 不可用且无缓存列表)")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Referer": "https://legulegu.com/",
    }
    frames, failed = [], 0
    for i, code in enumerate(codes):
        try:
            r = None
            for _ in range(2):  # one retry per page (legulegu throttles)
                r = requests.get(
                    f"https://legulegu.com/stockdata/index-composition?industryCode={code}",
                    headers=headers, timeout=10,
                )
                if r.status_code == 200:
                    break
                time.sleep(0.5)
            r.raise_for_status()
            frames.append(_parse_sw_cons_page(r.text))
        except Exception:  # noqa: BLE001 — skip broken pages, report later
            failed += 1
        if i == 9 and failed == 10:
            raise RuntimeError(
                "legulegu 连续 10 页失败 — 站点可能不可用,中止爬取")
        if (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(codes)} 页 ({failed} 失败)")
            if failed / (i + 1) > 0.5:
                raise RuntimeError(
                    f"申万三级成分页爬取失败过多 (前 {i + 1} 页失败 {failed})")
        time.sleep(0.15)  # be polite to legulegu
    print(f"  爬取完成: {len(codes)} 页, {failed} 页失败")
    if not frames or failed / len(codes) > 0.5:
        raise RuntimeError(f"申万三级成分页爬取失败过多 ({failed}/{len(codes)})")
    table = pl.concat(frames).unique(subset=["code"], keep="first")
    today = int(datetime.date.today().strftime("%Y%m%d"))
    return table.with_columns(pl.lit(today).alias("crawl_date"))


def _backfill_valuation_names(cache_dir: str = DEFAULT_VALUATION_CACHE_DIR) -> bool:
    """Fill an all-null ``name`` column in the valuation cache from Sina.

    Tushare ``daily_basic`` can succeed while ``stock_basic`` (the name source)
    is throttled, leaving names null.  Sina's spot is not rate-limited the same
    way, so it is used to backfill.  Returns True if the cache was rewritten.
    """
    cache = os.path.join(cache_dir, "sw_valuation.parquet")
    if not os.path.isfile(cache):
        return False
    table = pl.read_parquet(cache)
    if "name" not in table.columns or table["name"].null_count() != table.height:
        return False
    names = symbol_names(table["code"].to_list())
    names_df = pl.DataFrame({
        "code": list(names),
        "name": [names[c] for c in names],
    }).filter(pl.col("name") != "?")
    if names_df.is_empty():
        return False
    table = table.drop("name").join(names_df, on="code", how="left")
    table.write_parquet(cache)
    print("[data] 估值缓存 name 列已从新浪快照补齐")
    return True


def _load_sw_valuation_table(
    cache_dir: str = DEFAULT_VALUATION_CACHE_DIR,
    refresh: bool = False,
) -> pl.DataFrame:
    """Load the per-stock valuation table, preferring Tushare ``daily_basic``.

    A same-day cache is reused without any API call.  On a miss, Tushare
    ``daily_basic`` (one call, whole market) fills the cache; if that fails
    (no token / throttled / no permission), the legulegu 申万三级 crawl is
    the fallback.  If both fail and a stale cache exists, the stale cache is
    reused (matching the fallback style of the market-cap pool).
    """
    import datetime

    cache = os.path.join(cache_dir, "sw_valuation.parquet")
    today = int(datetime.date.today().strftime("%Y%m%d"))
    cached = None
    if not refresh and os.path.isfile(cache):
        try:
            df = pl.read_parquet(cache)
            if int(df["crawl_date"].max()) == today:
                print(f"[data] Using cached valuation table at '{cache}'")
                return df.drop("crawl_date")
            cached = df
        except Exception:  # noqa: BLE001 — treat a corrupt cache as absent
            pass
    try:
        table = ts_valuation_table(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        table.write_parquet(cache)
        _backfill_valuation_names(cache_dir)
        return table.drop("crawl_date")
    except Exception as e:
        print(f"[data] tushare daily_basic 不可用 ({type(e).__name__}: "
              f"{str(e)[:60]}) — 回退乐咕爬虫 …")
    try:
        print("[data] Crawling per-stock valuation (legulegu 申万三级, ~335 pages) …")
        table = _crawl_sw_valuation_table(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        table.write_parquet(cache)
        _backfill_valuation_names(cache_dir)
        return table.drop("crawl_date")
    except Exception as e:
        if cached is not None:
            print(f"[data] Valuation crawl failed ({type(e).__name__}) — "
                  f"reusing stale cache …")
            _backfill_valuation_names(cache_dir)
            return cached.drop("crawl_date")
        raise


def _parse_valuation_spec(spec: str) -> tuple[str, str, int]:
    """Parse a valuation spec ("pe_low:30" / "pe_ttm:10" / "pb:20" / "dy:15")."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Unrecognized valuation spec: {spec!r}")
    head, n = parts
    head_parts = head.rsplit("_", 1)
    if len(head_parts) == 2 and head_parts[1] in ("low", "high"):
        metric, direction = head_parts
    else:
        metric = head
        direction = _DEFAULT_DIRECTION.get(metric)
        if direction is None:
            raise ValueError(
                f"Unknown valuation metric {metric!r}; expected one of "
                f"{sorted(_VALUATION_METRICS)}")
    if metric not in _VALUATION_METRICS:
        raise ValueError(
            f"Unknown valuation metric {metric!r}; expected one of "
            f"{sorted(_VALUATION_METRICS)}")
    if direction not in ("low", "high"):
        raise ValueError(f"Unknown direction {direction!r}; expected 'low' or 'high'")
    return metric, direction, int(n)


def valuation_symbols(
    spec: str,
    cache_dir: str = DEFAULT_VALUATION_CACHE_DIR,
    refresh: bool = False,
) -> list[str]:
    """Return the N codes with the lowest/highest valuation metric.

    ``spec`` is ``"<metric>[:<direction>]:<N>"`` where ``metric`` is one of
    ``pe`` (静态市盈率), ``pe_ttm`` (市盈率TTM), ``pb`` (市净率), ``dy``
    (股息率) and ``direction`` is ``low`` (default; ``high`` for ``dy``).
    Only stocks with a positive metric are considered (loss-making stocks have
    no meaningful PE).  Requires network on the first call of the day (legulegu
    crawl, cached to parquet).

    Example:
        valuation_symbols("pe_low:30")  # 30 lowest static-PE stocks
    """
    metric, direction, n = _parse_valuation_spec(spec)
    table = _load_sw_valuation_table(cache_dir, refresh)
    col = metric
    if col not in table.columns:
        raise RuntimeError(
            f"Valuation column {col!r} missing from the table "
            f"(columns: {list(table.columns)})")
    sub = table.filter(pl.col(col).is_finite() & (pl.col(col) > 0))
    sub = sub.sort(col, descending=(direction == "high")).head(n)
    return sub["code"].to_list()


def _sina_market_caps(
    codes: list[str],
    cache_dir: str = DEFAULT_UNIVERSE_CACHE_DIR,
) -> dict[str, float]:
    """Total market cap (yuan) per code = latest close × outstanding_share.

    Uses the Sina daily source (``stock_zh_a_daily``), which is not subject to
    the Eastmoney ``push2`` rate-limit.  Per-stock, so this is slower than the
    Eastmoney spot; used only as a fallback.  Results are cached per day to
    ``cache_dir/sina_caps.parquet`` (only missing codes are fetched).  Failed
    lookups map to NaN.
    """
    import akshare as ak
    import time
    import datetime

    cache_file = os.path.join(cache_dir, "sina_caps.parquet")
    today = int(datetime.date.today().strftime("%Y%m%d"))
    cached: dict[str, float] = {}
    if os.path.isfile(cache_file):
        try:
            df = pl.read_parquet(cache_file)
            if int(df["crawl_date"].max()) == today:
                cached = {r["code"]: float(r["cap"])
                          for r in df[["code", "cap"]].to_dicts()}
        except Exception:  # noqa: BLE001 — treat a corrupt cache as absent
            pass

    missing = [c for c in codes if c not in cached]
    if missing:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y%m%d")
        for code in missing:
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            try:
                raw = ak.stock_zh_a_daily(
                    symbol=prefix + code, start_date=start, end_date=end, adjust="")
                last = raw.iloc[-1]
                cached[code] = float(last["close"] * last["outstanding_share"])
            except Exception:
                cached[code] = float("nan")
            time.sleep(0.05)
        os.makedirs(cache_dir, exist_ok=True)
        pl.DataFrame({
            "code": list(cached),
            "cap": [cached[c] for c in cached],
            "crawl_date": [today] * len(cached),
        }).write_parquet(cache_file)
    return {c: cached.get(c, float("nan")) for c in codes}


def market_cap_symbols(
    n: int,
    ascending: bool = False,
    float_cap: bool = False,
    cache_dir: str = DEFAULT_UNIVERSE_CACHE_DIR,
) -> list[str]:
    """Return the top/bottom ``n`` A-share codes by market cap.

    Requires network (akshare). Uses 总市值 (total market cap) by default, or
    流通市值 (free-float cap) if ``float_cap=True``.

    ``ascending=False`` → the ``n`` largest (large-cap pool);
    ``ascending=True`` → the ``n`` smallest (small-cap pool).

    Three-tier strategy:

      * Primary: Tushare ``daily_basic`` (``total_mv`` / ``circ_mv`` for the
        whole market in one call, cached per day via the valuation table).
      * Secondary: Eastmoney spot (``stock_zh_a_spot_em``) — one bulk call.
      * Fallback: when both are unavailable, the Sina daily source over the
        CSI300 universe (large-cap pool ⊆ CSI300), computing 总市值 =
        close × outstanding_share per stock.  The fallback is slower and only
        provides *total* cap (``float_cap`` is ignored), but it works when
        Eastmoney is down.
    """
    import math
    import akshare as ak

    col = "circ_cap" if float_cap else "cap"
    try:
        table = ts_valuation_table(DEFAULT_VALUATION_CACHE_DIR)
        cache = os.path.join(DEFAULT_VALUATION_CACHE_DIR, "sw_valuation.parquet")
        os.makedirs(DEFAULT_VALUATION_CACHE_DIR, exist_ok=True)
        table.write_parquet(cache)
        if col in table.columns:
            sub = table.filter(pl.col(col).is_finite() & (pl.col(col) > 0))
            sub = sub.sort(col, descending=not ascending).head(n)
            return sub["code"].to_list()
    except Exception:  # noqa: BLE001 — no token / throttled → next source
        pass

    col = "流通市值" if float_cap else "总市值"
    try:
        spot = _retry(lambda: ak.stock_zh_a_spot_em(), attempts=3, delay=1.5)
        df = spot.sort_values(col, ascending=ascending, na_position="last")
        return df["代码"].astype(str).head(n).tolist()
    except Exception:
        print("[data] Eastmoney spot unavailable — falling back to Sina daily "
              "(per-stock market cap over CSI300, ~30s–2min) …")
        codes = csi300_symbols(cache_dir=cache_dir)
        caps = _sina_market_caps(codes, cache_dir)
        valid = [(c, caps[c]) for c in codes if not math.isnan(caps[c])]
        valid.sort(key=lambda t: t[1], reverse=not ascending)
        return [c for c, _ in valid[:n]]


def resolve_symbols(spec) -> list[str]:
    """Resolve a stock-pool spec into a list of 6-digit codes.

    ``spec`` may be:

      * a list/tuple of codes (returned as-is, e.g. ``["600519", "000001"]``),
      * ``"csi300"`` or ``"csi300:30"`` — all / first N CSI300 constituents,
      * ``"industry:<keyword>"`` — stocks whose 证监会 industry or name
        contains the keyword (Tushare ``stock_basic``, daily cache),
      * ``"valuation:<metric>[:<dir>]:<N>"`` — top/bottom N by valuation
        (``pe``/``pe_ttm``/``pb``/``dy``, ``low`` default / ``high`` for dy),
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
    if spec.startswith("valuation:"):
        return valuation_symbols(spec.split(":", 1)[1])
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
        return f"行业(证监会) · {spec.split(':', 1)[1]}"
    if spec.startswith("valuation:"):
        parts = spec.split(":", 1)[1].split(":")
        head_parts = parts[0].rsplit("_", 1)
        if len(head_parts) == 2 and head_parts[1] in ("low", "high"):
            metric, direction = head_parts
        else:
            metric = parts[0]
            direction = _DEFAULT_DIRECTION.get(metric, "low")
        n = parts[-1]
        return (f"估值 · {_VALUATION_LABELS.get(metric, metric)} "
                f"{'最低' if direction == 'low' else '最高'}（前 {n}）")
    if spec.startswith("market_cap_small:"):
        return f"总市值最小（前 {spec.split(':', 1)[1]}）"
    if spec == "market_cap" or spec.startswith("market_cap:"):
        n = spec.split(":", 1)[1] if ":" in spec else "50"
        return f"总市值最大（前 {n}）"
    if "," in spec:
        return f"自选列表（{len(spec.split(','))} 只）"
    return f"单只 · {spec}"


def symbol_names(
    codes: list[str],
    universe_cache_dir: str = DEFAULT_UNIVERSE_CACHE_DIR,
    valuation_cache_dir: str = DEFAULT_VALUATION_CACHE_DIR,
) -> dict[str, str]:
    """Map 6-digit codes to stock names, from offline caches first.

    Name sources, in order (each covers the codes still missing):

      1. the Tushare ``stock_basic`` daily cache (fast, offline);
      2. the valuation cache's ``name`` column (fast, offline);
      3. the Sina spot source (akshare, ~20s, classic text progress bar).

    Unknown codes, or a failed lookup, map to ``"?"`` — this function never
    raises, so it can't abort a pipeline just because name decoration is
    unavailable.
    """
    import akshare as ak

    # Normalize input codes to the bare 6-digit form used as cache keys
    # ("600519.SH" and "sh600519" both → "600519").
    norm = {c: c.split(".")[0][-6:] if "." in c else c[-6:] for c in codes}
    names: dict[str, str] = {}
    for cache in (os.path.join(universe_cache_dir, "stock_basic.parquet"),
                  os.path.join(valuation_cache_dir, "sw_valuation.parquet")):
        if os.path.isfile(cache):
            try:
                df = pl.read_parquet(cache)
                if "name" in df.columns:
                    names.update({
                        r["code"]: r["name"]
                        for r in df.select(["code", "name"]).to_dicts()
                        if r["name"]
                    })
            except Exception:  # noqa: BLE001 — a broken cache is skipped
                pass

    missing = [c for c in codes if norm[c] not in names]
    if missing:
        # akshare's ``get_tqdm`` switches to the ipywidgets ``tqdm.notebook``
        # inside a Jupyter kernel; executed headlessly (nbconvert) that leaves
        # a dangling widget-view output ("Error displaying widget").  Force
        # the classic text progress bar instead.
        import tqdm
        import tqdm.notebook as _tn

        prev = _tn.tqdm
        _tn.tqdm = tqdm.tqdm
        try:
            spot = _retry(lambda: ak.stock_zh_a_spot(), attempts=2, delay=1.0)
        except Exception:
            spot = None
        finally:
            _tn.tqdm = prev
        if spot is not None:
            name_map = {
                code[-6:]: name
                for code, name in zip(spot["代码"].astype(str),
                                      spot["名称"].astype(str))
            }
            names.update({c: name_map[norm[c]] for c in missing
                          if norm[c] in name_map})
    return {c: names.get(norm[c], "?") for c in codes}


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
