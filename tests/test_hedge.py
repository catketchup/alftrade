"""Tests for hedge.py: HedgeManager, FuturesContract, compute_hedged_equity."""

import numpy as np
import pytest

from alftrade.hedge import HedgeManager, FuturesContract, compute_hedged_equity


# ── Helpers ────────────────────────────────────────────────────────────

def _pre_feed(hm: HedgeManager, n: int = 30, seed: int = 1, beta_true: float = 1.0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        mkt = rng.normal(0.0, 0.01)
        port = beta_true * mkt + rng.normal(0.0, 0.003)
        hm._port_rets.append(port)
        hm._mkt_rets.append(mkt)
    hm._update_beta()


# ── FuturesContract ────────────────────────────────────────────────────

def test_futures_contract_defaults():
    c = FuturesContract("IM")
    assert c.symbol == "IM"
    assert c.multiplier == 200.0
    assert c.roll_cost_annual_bps == 0.06


# ── beta estimation ────────────────────────────────────────────────────

def test_beta_fallback_no_data():
    hm = HedgeManager()
    hm._update_beta()
    assert hm.beta == 1.0
    assert hm.beta_t == 0.0


def test_beta_recovers_known():
    hm = HedgeManager(beta_window=200)
    rng = np.random.default_rng(99)
    for _ in range(300):
        mkt = rng.normal(0.0, 0.01)
        port = 1.3 * mkt + rng.normal(0.0, 0.002)
        hm._port_rets.append(port)
        hm._mkt_rets.append(mkt)
    hm._update_beta()
    assert 1.25 <= hm.beta <= 1.35
    assert hm.beta_t > 5  # very significant


# ── should_hedge ──────────────────────────────────────────────────────

def test_should_hedge_strong():
    hm = HedgeManager(min_beta_t=2.0)
    hm._beta = 1.3
    hm._beta_t = 3.5
    assert hm.should_hedge


def test_should_hedge_weak():
    hm = HedgeManager(min_beta_t=2.0)
    hm._beta = 1.3
    hm._beta_t = 0.8
    assert not hm.should_hedge


# ── step / lot calculation ────────────────────────────────────────────

def test_step_no_hedge_when_insignificant():
    hm = HedgeManager(min_beta_t=100)
    _pre_feed(hm, beta_true=1.0)
    pnl, lots, beta = hm.step(1, 1_000_000, 0.005, 0.003, 5000)
    assert lots == 0
    assert pnl == 0.0


def test_step_opens_hedge():
    hm = HedgeManager(beta_window=40, min_beta_t=0.0)
    _pre_feed(hm, beta_true=1.2, n=50)
    pnl, lots, beta = hm.step(1, 1_000_000, 0.012, 0.01, 5600)
    assert lots >= 0
    assert 0.8 < beta < 1.6


def test_step_settles_yesterday():
    hm = HedgeManager(min_beta_t=0.0)
    _pre_feed(hm, beta_true=1.0)
    pnl1, lots1, _ = hm.step(1, 1_000_000, 0.01, 0.01, 5000)
    assert lots1 > 0
    assert pnl1 == 0.0
    # Day 2: market drops 1 % → short makes money
    pnl2, _, _ = hm.step(2, 1_000_000, -0.01, -0.01, 4950)
    assert pnl2 > 0
    assert hm.cumulative_pnl > 0


def test_cumulative_pnl_tracks_settlements():
    hm = HedgeManager(min_beta_t=0.0)
    _pre_feed(hm, beta_true=1.0)
    hm.step(1, 1_000_000, 0.01, 0.01, 5000)
    before = hm.cumulative_pnl
    hm.step(2, 1_000_000, 0.02, 0.02, 5100)  # rally → short loses
    assert hm.cumulative_pnl < before


def test_max_leverage_caps():
    hm = HedgeManager(min_beta_t=0.0, max_leverage=0.5)
    _pre_feed(hm, beta_true=2.0)
    # β=2, equity=1M → raw notional=2M, capped at 0.5M
    # price=5000 × 200 = 1M/lot → 0 lots
    pnl, lots, _ = hm.step(1, 1_000_000, 0.02, 0.01, 5000)
    assert lots == 0


# ── reset ─────────────────────────────────────────────────────────────

def test_reset():
    hm = HedgeManager()
    _pre_feed(hm, beta_true=1.0)
    hm._lots = 3
    hm._cumulative_pnl = 5000.0
    hm._index_price = 5600.0
    hm.reset()
    assert hm.lots == 0
    assert hm.cumulative_pnl == 0.0
    assert hm._index_price == 0.0
    assert hm.beta == 1.0
    assert hm.beta_t == 0.0


# ── compute_hedged_equity ─────────────────────────────────────────────

def test_compute_hedged_equity():
    equity = [1000.0, 1010.0, 1020.0]
    hedge_pnl = [5.0, -3.0, 10.0]
    hedged = compute_hedged_equity(equity, [], hedge_pnl)
    assert hedged[0] == pytest.approx(1000 + 5)
    assert hedged[1] == pytest.approx(1010 + 5 - 3)
    assert hedged[2] == pytest.approx(1020 + 5 - 3 + 10)


def test_compute_hedged_equity_empty():
    assert compute_hedged_equity([], [], []) == []
