"""
ts_source.py - Optional Tushare-backed data source.

Tushare pro requires a token, configured via the ``TUSHARE_TOKEN``
environment variable or ``data/tushare_token.txt`` (see ``_load_token``),
and throttles most interfaces hard at low point tiers (``stock_basic`` and
``daily_basic`` degrade to 1 call/hour when abused).  Every result is therefore
cached per day, mirroring the legulegu / Sina caches in ``datasource``.

Interfaces used (permission is tier-dependent):

  * ``stock_basic`` — whole-market basic info (name, 证监会 ``industry``,
    ``market``, ``area``) in one call;
  * ``daily_basic`` — whole-market daily valuation (``pe``, ``pe_ttm``,
    ``pb``, ``dv_ttm``, ``total_mv``) for one trade date;
  * ``trade_cal``   — trading calendar, to find the latest trade date.

Functions in this module raise a clear error when ``TUSHARE_TOKEN`` is unset
or an interface is throttled, so callers can fall back to other sources.
"""

import os
from typing import Optional

import polars as pl

DEFAULT_TS_CACHE_DIR = "data/universe"


def _load_token(project_dir: str = ".") -> str:
    """Resolve the tushare token: env var first, then a project-local file.

    Priority:

      1. ``TUSHARE_TOKEN`` environment variable (per-process override);
      2. ``<project_dir>/data/tushare_token.txt`` — plain text, one line.
         ``data/`` is gitignored, and the paper cron job ``cd``s into the
         project dir, so this works for scheduled runs too.

    Returns ``""`` when neither is configured.
    """
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    path = os.path.join(project_dir, "data", "tushare_token.txt")
    try:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError:
        token = ""
    return token


def _pro():
    """Lazily build the Tushare pro client from ``_load_token()``."""
    token = _load_token()
    if not token:
        raise RuntimeError(
            "未找到 tushare token,二选一配置:\n"
            "  1) export TUSHARE_TOKEN=<你的token>   (仅当前 shell 生效)\n"
            "  2) echo <你的token> > data/tushare_token.txt  "
            "(项目本地,已被 .gitignore 忽略)")
    import tushare as ts

    return ts.pro_api(token)


def _today_int() -> int:
    import datetime

    return int(datetime.date.today().strftime("%Y%m%d"))


def _stock_basic_table(cache_dir: str = DEFAULT_TS_CACHE_DIR) -> pl.DataFrame:
    """Whole-market basic info (code/name/industry/market/area), cached per day.

    One ``stock_basic`` call covers the full market (5562 listed stocks);
    results are cached to ``cache_dir/stock_basic.parquet``.  On a same-day
    cache hit no API call is made (important: ``stock_basic`` is heavily
    throttled at low tiers).  If the call fails and a stale cache exists, the
    stale cache is reused.
    """
    import datetime

    cache = os.path.join(cache_dir, "stock_basic.parquet")
    today = _today_int()
    cached: Optional[pl.DataFrame] = None
    if os.path.isfile(cache):
        try:
            df = pl.read_parquet(cache)
            if int(df["crawl_date"].max()) == today:
                return df.drop("crawl_date")
            cached = df
        except Exception:  # noqa: BLE001 — treat a corrupt cache as absent
            pass
    try:
        raw = _pro().stock_basic(
            exchange="", list_status="L",
            fields="ts_code,name,industry,market,area")
        table = (
            pl.from_pandas(raw)
            .with_columns([
                pl.col("ts_code").str.slice(0, 6).alias("code"),
                pl.lit(today).alias("crawl_date"),
            ])
            .select(["code", "name", "industry", "market", "area", "crawl_date"])
        )
        os.makedirs(cache_dir, exist_ok=True)
        table.write_parquet(cache)
        return table.drop("crawl_date")
    except Exception as e:
        if cached is not None:
            print(f"[data] tushare stock_basic 不可用 "
                  f"({type(e).__name__}: {str(e)[:60]}) — 复用陈旧缓存 …")
            return cached.drop("crawl_date")
        raise


def ts_industry_symbols(
    keyword: str,
    cache_dir: str = DEFAULT_TS_CACHE_DIR,
) -> list[str]:
    """Return the codes whose 证监会 industry or name contains ``keyword``.

    Tushare at most tiers exposes no industry-constituent interface (THS/SW
    boards need higher permission), so the industry pool is built from the
    ``stock_basic.industry`` field (证监会行业分类, coarse — e.g. all liquor
    is "酒、饮料和精制茶制造业") plus a name match.

    Example:
        ts_industry_symbols("酒")   # every listed liquor/brewery/soft-drink stock
        ts_industry_symbols("白酒")  # only names containing 白酒 (老白干酒 etc.)
    """
    table = _stock_basic_table(cache_dir)
    kw = keyword.strip()
    m = table.filter(
        pl.col("industry").str.contains(kw, literal=True)
        | pl.col("name").str.contains(kw, literal=True)
    )
    if m.is_empty():
        raise ValueError(
            f"tushare 证监会行业与股票名称中均无 '{kw}' — 证监会分类较粗"
            f"(如 '酒、饮料和精制茶制造业'),试试更短的关键词,例如 industry:酒")
    return sorted(m["code"].to_list())


def ts_valuation_table(
    cache_dir: str = "data/valuation",
    universe_cache_dir: str = DEFAULT_TS_CACHE_DIR,
) -> pl.DataFrame:
    """Whole-market valuation from ``daily_basic`` → canonical cache schema.

    One ``daily_basic`` call per trade date returns pe / pe_ttm / pb /
    dv_ttm / total_mv for every listed stock.  Units are converted to the
    canonical cache schema (``cap`` in 亿 yuan, ``dy`` in %); stock names are
    joined from the ``stock_basic`` cache when available.  ``crawl_date`` is
    set to today.
    """
    import datetime

    pro = _pro()
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=10)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    open_dates = cal[cal["is_open"] == 1]["cal_date"]
    if open_dates.empty:
        raise RuntimeError("tushare 交易日历近期无开市日")
    last_trade_date = open_dates.max()

    raw = pro.daily_basic(
        trade_date=last_trade_date,
        fields="ts_code,pe,pe_ttm,pb,dv_ttm,total_mv,circ_mv")
    table = (
        pl.from_pandas(raw)
        .with_columns(
            pl.col("ts_code").str.slice(0, 6).alias("code"),
            (pl.col("total_mv") / 1e4).alias("cap"),    # 万元 → 亿元
            (pl.col("circ_mv") / 1e4).alias("circ_cap"),
            pl.col("dv_ttm").alias("dy"),
        )
        .select(["code", "pe", "pe_ttm", "pb", "dy", "cap", "circ_cap"])
    )

    names: Optional[pl.DataFrame] = None
    try:
        names = _stock_basic_table(universe_cache_dir).select(["code", "name"])
    except Exception:  # noqa: BLE001 — names are decoration, not essential
        pass
    if names is not None:
        table = table.join(names, on="code", how="left")
    else:
        table = table.with_columns(pl.lit(None).cast(pl.String).alias("name"))
    return table.with_columns(pl.lit(_today_int()).alias("crawl_date")).select(
        ["code", "name", "pe", "pe_ttm", "pb", "dy", "cap", "circ_cap",
         "crawl_date"])
