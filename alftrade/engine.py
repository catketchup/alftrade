"""
engine.py - Cross-sectional trading engine that consumes daily bars and
produces per-stock signals.

This is the daily, multi-stock analogue of hftrade's ``TradingEngine.on_tick``.
The data flow is:

    DailyBar[s]  →  TradingEngine.on_cross_section(date, bars)
                 →  FactorCalculator[s].update(bar)
                 →  cross-sectional standardization
                 →  Signal[s]  (one list per stock)

Two signal-generation modes (same split as hftrade):

  1. Raw mode (default): each factor is independently turned into a Signal
     via threshold rules. Useful for inspection and before calibration.

  2. Calibrated mode: a ``CalibrationResult`` (from ``FactorCalibrator.fit``)
     is supplied. Signal strength = Σ β_k · z(factor_k) — the *predicted
     forward return* for that stock, unified across all factors. The
     portfolio manager then ranks stocks cross-sectionally by this predicted
     return and longs the top names.

Cross-sectional standardization
-------------------------------
A key difference from hftrade: in the calibrated path we z-score each factor
*within the day's cross-section* before applying β. This matches what
``FactorCalibrator.fit`` does during training (see calibration.py) and is
standard practice for cross-sectional equity models — it removes cross-day
scale drift and makes β interpretable as "per-1σ cross-sectional exposure".

Long-only convention
--------------------
A-share cash accounts cannot short. Therefore signals are emitted with
direction ∈ {+1, 0}: stocks with positive predicted return are LONG
candidates, stocks with non-positive predicted return are FLAT (the portfolio
manager will not short them, only drop them from the book).

Execution convention (simplification)
--------------------------------------
Signals computed from day t's close produce trades that execute at day t's
close price. This is a pedagogical simplification (a real A-share backtest
would execute at t+1's open with t's signals). Because the system rebalances
at most once per day per stock, the A-share T+1 rule is naturally satisfied:
you never buy and sell the same name within one bar. See ``portfolio.py`` for
the T+1 enforcement on the position side.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from alftrade.data import DailyBar
from alftrade.factors import FactorCalculator, FACTOR_NAMES, compute_cross_section

if TYPE_CHECKING:
    from alftrade.calibration import CalibrationResult


class Signal:
    """A trading signal for one stock on one date.

    Attributes:
        symbol: Ticker the signal applies to.
        name: Factor / model that produced it (e.g. "momentum_5d" or "model").
        direction: +1 = LONG, 0 = FLAT. Never -1 in this long-only system.
        strength: Magnitude. In calibrated mode this is the predicted forward
                  return; in raw mode it is the factor value (or a ratio).
    """

    def __init__(self, symbol: str, name: str, direction: int, strength: float):
        self.symbol = symbol
        self.name = name
        self.direction = direction
        self.strength = strength

    def __repr__(self):
        arrow = "LONG" if self.direction > 0 else "FLAT"
        return f"Signal({self.symbol} {self.name}: {arrow}, strength={self.strength:.5f})"


@dataclass
class CrossSection:
    """One day's worth of data + computed factors.

    Bundles the bars, per-symbol factors, and the cross-sectionally
    standardized factor matrix so that downstream consumers (engine,
    portfolio manager) don't recompute them.
    """
    date: int
    bars: dict[str, DailyBar]
    factors_by_symbol: dict[str, dict[str, float]]
    z_matrix: np.ndarray            # (n_stocks, n_factors) standardized
    symbols: list[str]              # row order matches z_matrix
    factor_names: list[str]         # column order matches z_matrix


def standardize_cross_section(
    factors_by_symbol: dict[str, dict[str, float]],
    factor_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Z-score each factor across the day's universe of stocks.

    Returns the (n_stocks, n_factors) standardized matrix and the row
    ordering of symbols. Constant factors (std ≈ 0) are mapped to 0 so they
    carry no cross-sectional information that day.

    This must be applied identically in ``FactorCalibrator.fit`` (offline)
    and in ``TradingEngine._calibrated_signals`` (online) so β is applied to
    the same factor representation in both phases.
    """
    symbols = list(factors_by_symbol.keys())
    n, k = len(symbols), len(factor_names)
    X = np.zeros((n, k))
    for i, sym in enumerate(symbols):
        f = factors_by_symbol[sym]
        for j, name in enumerate(factor_names):
            X[i, j] = f.get(name, 0.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    return Z, symbols


class TradingEngine:
    """Processes one day's cross-section of bars into per-stock signals.

    Owns the per-symbol ``FactorCalculator`` instances (so rolling state
    persists across days). Call ``on_cross_section`` once per trading day
    with that day's ``{symbol: DailyBar}`` map.

    Args:
        factor_names: Ordered factors to use. Defaults to all 10 in
            ``FACTOR_NAMES``. Calibration β and these names must match.
        calibration: Optional ``CalibrationResult`` for calibrated mode.
        raw_thresholds: Per-factor threshold rules for raw mode. If a factor
            is absent from this dict, its raw-mode rule is skipped.
            Example: ``{"momentum_5d": 0.02}`` emits a LONG signal when
            momentum_5d > 0.02 and a FLAT signal when < -0.02.
    """

    def __init__(
        self,
        factor_names: Optional[list[str]] = None,
        calibration: "Optional[CalibrationResult]" = None,
        raw_thresholds: Optional[dict[str, float]] = None,
    ):
        self.factor_names = list(factor_names or FACTOR_NAMES)
        self.calibration = calibration
        self.raw_thresholds = raw_thresholds or {
            # Mild defaults so raw mode produces *some* signals for demo use.
            "momentum_5d": 0.02,
            "rsi_14": 0.30,        # rsi is on [-1,1] scale after rescaling
            "ma_dev_20": 0.02,
        }
        self._calcs: dict[str, FactorCalculator] = {}

    def on_cross_section(self, date: int, bars: dict[str, DailyBar]) -> dict[str, list[Signal]]:
        """Process one date's bars and return {symbol: [Signal]}.

        Drops suspended symbols (no factors, no signals). Advances each
        symbol's rolling factor state as a side effect.
        """
        tradeable = {s: b for s, b in bars.items() if not b.is_suspended}
        factors_by_symbol = compute_cross_section(tradeable, self._calcs)
        if not factors_by_symbol:
            return {}

        if self.calibration is not None:
            return self._calibrated_signals(factors_by_symbol)
        return self._raw_signals(factors_by_symbol)

    def build_cross_section(self, date: int, bars: dict[str, DailyBar]) -> CrossSection:
        """Compute factors + standardized matrix without generating signals.

        Used by the calibrator/profiler during the offline phase so they
        share the exact same factor pipeline as the live engine.
        """
        tradeable = {s: b for s, b in bars.items() if not b.is_suspended}
        factors_by_symbol = compute_cross_section(tradeable, self._calcs)
        Z, symbols = standardize_cross_section(factors_by_symbol, self.factor_names)
        return CrossSection(
            date=date,
            bars=tradeable,
            factors_by_symbol=factors_by_symbol,
            z_matrix=Z,
            symbols=symbols,
            factor_names=list(self.factor_names),
        )

    def _raw_signals(self, factors_by_symbol: dict[str, dict[str, float]]) -> dict[str, list[Signal]]:
        """Threshold-based per-factor signals (one Signal per firing factor)."""
        out: dict[str, list[Signal]] = {}
        for sym, f in factors_by_symbol.items():
            sigs: list[Signal] = []
            for name in self.factor_names:
                thr = self.raw_thresholds.get(name)
                if thr is None:
                    continue
                value = f.get(name, 0.0)
                if value > thr:
                    sigs.append(Signal(sym, name, 1, value))
                elif value < -thr:
                    # Long-only: a negative factor reading is a FLAT signal
                    # (we don't short). Still emitted so the aggregator can
                    # see it as a vote against going long.
                    sigs.append(Signal(sym, name, 0, abs(value)))
            if sigs:
                out[sym] = sigs
        return out

    def _calibrated_signals(self, factors_by_symbol: dict[str, dict[str, float]]) -> dict[str, list[Signal]]:
        """Predicted-return signals using cross-sectionally standardized factors.

        For each stock:
            predicted_return = Σ_k β_k · z_score(factor_k)
            direction = +1 (LONG) if predicted_return > 0 else 0 (FLAT)
            strength  = |predicted_return|

        The signal is emitted under the synthetic name "model" so the
        aggregator treats the whole factor model as one vote. (This differs
        from raw mode, which emits one signal per factor.)
        """
        from alftrade.calibration import CalibrationResult

        cal: CalibrationResult = self.calibration
        Z, symbols = standardize_cross_section(factors_by_symbol, self.factor_names)
        betas = np.array([cal.betas.get(name, 0.0) for name in self.factor_names])
        pred = Z @ betas

        out: dict[str, list[Signal]] = {}
        for i, sym in enumerate(symbols):
            p = float(pred[i])
            direction = 1 if p > 0 else 0
            if abs(p) > 1e-9:
                out[sym] = [Signal(sym, "model", direction, abs(p))]
        return out

    def reset(self) -> None:
        """Drop all per-symbol factor calculators (start a new backtest)."""
        self._calcs.clear()
