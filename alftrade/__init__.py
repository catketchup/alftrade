"""alftrade — A-share low-frequency cross-sectional trading system.

The package mirrors the layered architecture of ``hftrade`` (a tick-level
single-instrument high-frequency engine) but adapts every layer to the
A-share daily-frequency cross-sectional setting:

    hftrade (tick, single stock)          alftrade (daily, multi stock)
    ------------------------------------------------------------------
    Tick                                  DailyBar
    PyFactorCalculator (C++/Cython)       compute_factor_panel (polars vectorized)
    TradingEngine.on_tick → [Signal]      TradingEngine.on_cross_section → {sym: [Signal]}
    SignalAggregator (per tick)           SignalAggregator (per stock)
    PositionManager (single stock)        PortfolioManager (cross-section rebalance)
    RiskManager (tick risk)               RiskManager (T+1 / price-limit / lot / concentration)
    FactorCalibrator (time-series Ridge)  FactorCalibrator (Fama-MacBeth cross-sectional Ridge)
    FactorProfiler                        FactorProfiler (pooled IC, quantile returns, FM t-stats)

Signal generation runs at most once per trading day per stock.  Factors are
computed once in a vectorized polars pass over the parquet-read panel, then
looked up per date by the engine.
"""

from alftrade.data import DailyBar, StockUniverse, generate_simulated_market, bars_to_frame
from alftrade.factors import FACTOR_NAMES, compute_factor_panel, factor_panel_to_dict
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
from alftrade.store import (
    write_market_parquet,
    ParquetPanel,
    load_market_parquet,
    load_factor_panel,
    load_market_with_factors,
)

__all__ = [
    "DailyBar",
    "StockUniverse",
    "generate_simulated_market",
    "bars_to_frame",
    "FACTOR_NAMES",
    "compute_factor_panel",
    "factor_panel_to_dict",
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
    "write_market_parquet",
    "ParquetPanel",
    "load_market_parquet",
    "load_factor_panel",
    "load_market_with_factors",
]
