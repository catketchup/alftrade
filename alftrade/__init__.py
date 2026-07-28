"""alftrade — A-share low-frequency cross-sectional trading system.

The package mirrors the layered architecture of ``hftrade`` (a tick-level
single-instrument high-frequency engine) but adapts every layer to the
A-share daily-frequency cross-sectional setting:

    hftrade (tick, single stock)          alftrade (daily, multi stock)
    ------------------------------------------------------------------
    Tick                                  DailyBar
    PyFactorCalculator (C++/Cython)       FactorCalculator (numpy)
    TradingEngine.on_tick → [Signal]      TradingEngine.on_cross_section → {sym: [Signal]}
    SignalAggregator (per tick)           SignalAggregator (per stock)
    PositionManager (single stock)        PortfolioManager (cross-section rebalance)
    RiskManager (tick risk)               RiskManager (T+1 / price-limit / lot / concentration)
    FactorCalibrator (time-series Ridge)  FactorCalibrator (Fama-MacBeth cross-sectional Ridge)
    FactorProfiler                        FactorProfiler (pooled IC, quantile returns, FM t-stats)

Signal generation runs at most once per trading day per stock.
"""

from alftrade.data import DailyBar, StockUniverse, generate_simulated_market
from alftrade.factors import FactorCalculator, FACTOR_NAMES
from alftrade.engine import Signal, TradingEngine
from alftrade.portfolio import (
    Position,
    Trade,
    Portfolio,
    CompositeSignal,
    SignalAggregator,
    CostModel,
    PortfolioManager,
)
from alftrade.risk import RiskManager, RiskConfig
from alftrade.calibration import CalibrationResult, FactorCalibrator
from alftrade.profiler import FactorProfile, ModelProfile, FactorProfiler
from alftrade.hedge import HedgeManager, FuturesContract, compute_hedged_equity

__all__ = [
    "DailyBar",
    "StockUniverse",
    "generate_simulated_market",
    "FactorCalculator",
    "FACTOR_NAMES",
    "Signal",
    "TradingEngine",
    "Position",
    "Trade",
    "Portfolio",
    "CompositeSignal",
    "SignalAggregator",
    "CostModel",
    "PortfolioManager",
    "RiskManager",
    "RiskConfig",
    "CalibrationResult",
    "FactorCalibrator",
    "FactorProfile",
    "ModelProfile",
    "FactorProfiler",
    "HedgeManager",
    "FuturesContract",
    "compute_hedged_equity",
]
