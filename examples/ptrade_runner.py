# -*- coding: utf-8 -*-
"""
ptrade_runner.py — ptrade 侧下单策略模板(与 alftrade 的订单文件桥接)。

用法
====
1. 把本文件内容粘贴到 ptrade 的"策略"编辑器里(或由 ptrade 导入);
2. 修改下方 CONFIG 里的 WATCH_DIR,指向 alftrade 导出的订单目录
   (``data/orders``)在 ptrade 环境里可见的路径;
3. 启动策略。每个交易日 09:26 自动读取最新未执行的 orders.csv 并市价下单;
   已执行的计划日期记录在 DONE_FILE 里,绝不重复下单。

文件如何从 alftrade 传到 ptrade 环境(两种方式任选)
==================================================
alftrade 每天收盘(约 15:00 后)运行一次,把当日计划写到
``data/orders/<计划日期>/orders.csv``。ptrade 是托管在券商的封闭环境,
一般读不到你本地磁盘,需要把 orders 目录同步过去:

方式一:共享盘 / 网盘自动同步(推荐)
  把 ``data/orders`` 通过同步工具(坚果云/网盘客户端/rsync 定时任务)
  同步到 ptrade 环境可见的目录,例如 ``Z:\\alftrade_orders``;
  然后 WATCH_DIR = r"Z:\\alftrade_orders"。同步延迟约 1 分钟,
  09:26 执行时前一日收盘的文件早已就位。

方式二:每日手动拷贝
  每天 15:00 后把最新的 ``orders.csv`` 手动上传到 ptrade 环境;
  WATCH_DIR 指向该上传目录即可。

时间线
======
  T 日 15:00   alftrade 收盘:生成 T 日计划(在 T+1 开盘执行),写入
               data/orders/<T>/orders.csv
  T+1 09:26    ptrade 本策略:读取并执行(市价单,约 09:30 成交)
  T+1 15:00    alftrade 收盘:生成 T+1 计划(影子模拟在本地照常记账,
               与实盘成交允许存在滑点差异)

注意
====
* ptrade API 版本差异:不同券商的 ptrade 内置函数名可能不同(常见为
  ``order_shares(security, amount)`` 市价按股数下单);下单前请对照你环境
  里的文档核对函数名与参数。
* 集合竞价(9:15-9:25)通常不接收市价单,09:26 之后下单走连续竞价;
  如需参与开盘集合竞价,可把 run_daily 时间改到 09:24 并改用限价单
  (ptrade 的 order() 函数),但 9:20-9:25 不可撤单,风险自担。
* 买入数量已由 alftrade 按 100 股整手取整;卖出允许零股。
* 防重:以"计划日期"为单位记录已执行清单(DONE_FILE),重复启动策略
  不会重复下单。
"""

# ── 配置区 ────────────────────────────────────────────────────────────
CONFIG = {
    # alftrade 订单目录在 ptrade 环境里的路径(见文件顶部"文件如何传递")
    "WATCH_DIR": r"Z:\\alftrade_orders",
    # 已执行计划日期的记录文件(ptrade 环境内任意可写路径)
    "DONE_FILE": r"D:\\ptrade_done\\executed_dates.txt",
    # 每日执行时间(策略平台时间)
    "RUN_TIME": "09:26",
    # 单日最多下单笔数(防文件异常导致的连环下单)
    "MAX_ORDERS_PER_DAY": 10,
}

# ── ptrade 接口(按你的券商文档核对/替换) ──────────────────────────────
# from ptrade.api import order_shares, get_position, get_portfolio  # noqa

import csv
import os
import datetime


def initialize(context):
    context.done_dates = _load_done(CONFIG["DONE_FILE"])


def run_daily(context):
    today = datetime.date.today().strftime("%Y%m%d")
    for plan_dir in sorted(os.listdir(CONFIG["WATCH_DIR"]), reverse=True):
        plan_date = plan_dir
        if plan_date in context.done_dates:
            continue
        if plan_date >= today:
            continue  # 只执行"过去"的计划日
        orders_path = os.path.join(CONFIG["WATCH_DIR"], plan_dir, "orders.csv")
        if not os.path.isfile(orders_path):
            continue
        _execute(context, plan_date, orders_path)
        break  # 每次只补最近一个未执行的计划日


def _execute(context, plan_date, orders_path):
    log = []
    with open(orders_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) > CONFIG["MAX_ORDERS_PER_DAY"]:
        log.append(f"SKIP {plan_date}: {len(rows)} 笔超过单日上限")
        _append_log(log)
        _mark_done(context, plan_date)
        return
    for r in rows:
        symbol = r["symbol"]
        side = r["side"]
        qty = int(r["quantity"])
        try:
            # ── 下单前校验(函数名以你的 ptrade 文档为准) ──
            # if side == "BUY":
            #     needed = qty * float(r["ref_close"]) * 1.2
            #     if get_portfolio().cash < needed:
            #         raise RuntimeError("资金不足")
            # else:
            #     pos = get_position(symbol)  # 可卖数量(含 T+1 约束)
            #     if pos is None or pos.available_qty < qty:
            #         raise RuntimeError("可卖数量不足")
            # ── 市价下单 ──
            # order_shares(symbol, qty if side == "BUY" else -qty)
            log.append(f"OK {symbol} {side} {qty}")
        except Exception as e:  # noqa: BLE001
            log.append(f"FAIL {symbol} {side} {qty}: {e}")
    _append_log(log)
    _mark_done(context, plan_date)


def _load_done(path):
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def _mark_done(context, plan_date):
    context.done_dates.add(plan_date)
    with open(CONFIG["DONE_FILE"], "a") as f:
        f.write(plan_date + "\n")


def _append_log(lines):
    d = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(os.path.dirname(CONFIG["DONE_FILE"]) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(CONFIG["DONE_FILE"]),
                           f"execution_{d}.log"), "a") as f:
        f.write(datetime.datetime.now().strftime("%H:%M:%S")
                + " | " + " | ".join(lines) + "\n")


# 在 ptrade 里启用定时任务(把 run_daily 注册为每日回调;接口名以文档为准):
# run_daily(run_daily, time=CONFIG["RUN_TIME"])
