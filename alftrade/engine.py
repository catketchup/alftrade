"""
engine.py - Cross-sectional trading engine that consumes precomputed daily
factors and produces per-stock signals.

This is the daily, multi-stock analogue of hftrade's ``TradingEngine.on_tick``.
Factors are computed *outside* the engine (by ``factors.compute_factor_panel``
over the parquet-read panel) and handed in as a ``{date: {symbol: factors}}``
mapping; the engine looks up a day's factors, standardizes them
cross-sectionally, and emits signals. The data flow is:

    parquet panel ──(polars)──> factor panel ──> {date: {symbol: factors}}
                                                         ↓
    TradingEngine.on_cross_section(date, bars) ──> cross-sectional standardization
                                                         ↓
                                                Signal[s]  (one list per stock)

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
from alftrade.factors import FACTOR_NAMES

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
    neutralize: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Z-score each factor across the day's universe of stocks.

    Returns the (n_stocks, n_factors) standardized matrix and the row
    ordering of symbols. Constant factors (std ≈ 0) are mapped to 0 so they
    carry no cross-sectional information that day.

    This must be applied identically in ``FactorCalibrator.fit`` (offline)
    and in ``TradingEngine._calibrated_signals`` (online) so β is applied to
    the same factor representation in both phases.

    If ``neutralize`` is provided (e.g. ["market_beta"]), each target factor
    is residualised against the neutralise factors *before* z-scoring.  This
    strips out the component of a factor that can be explained by the
    neutralise variables — classic market / industry / size neutralisation.
    """
    symbols = list(factors_by_symbol.keys())
    n, k = len(symbols), len(factor_names)
    X = np.zeros((n, k))
    for i, sym in enumerate(symbols):
        f = factors_by_symbol[sym]
        for j, name in enumerate(factor_names):
            X[i, j] = f.get(name, 0.0)

    # ── Optional neutralisation ──
    if neutralize:
        neut_cols = [factor_names.index(n) for n in neutralize if n in factor_names]
        if neut_cols:
            X_neut = X[:, neut_cols]
            n_neut = len(neut_cols)
            for j in range(k):
                if j in neut_cols:
                    continue
                y = X[:, j]
                # Ridge L2 (tiny α = 1e-6) avoids singularity when neut cols
                # are collinear; equivalent to OLS for well-conditioned data.
                XtX = X_neut.T @ X_neut
                Xty = X_neut.T @ y
                coeff = np.linalg.solve(XtX + 1e-6 * np.eye(n_neut), Xty)
                X[:, j] = y - X_neut @ coeff

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    return Z, symbols


class TradingEngine:
    """Processes one day's cross-section of factors into per-stock signals.

    Factors are precomputed outside the engine (via polars) and supplied as a
    ``factors`` mapping ``{date: {symbol: {factor: value}}}``. Call
    ``on_cross_section`` once per trading day with that day's
    ``{symbol: DailyBar}`` map (bars are only used for suspension filtering
    and, via ``build_cross_section``, to carry prices into calibration).

    Args:
        factors: Precomputed factors ``{date: {symbol: {factor: value}}}``
            (see ``factors.factor_panel_to_dict`` / ``store.load_factor_panel``).
        factor_names: Ordered factors to use. Defaults to all in
            ``FACTOR_NAMES``. Calibration β and these names must match.
        calibration: Optional ``CalibrationResult`` for calibrated mode.
        raw_thresholds: Per-factor threshold rules for raw mode. If a factor
            is absent from this dict, its raw-mode rule is skipped.
            Example: ``{"momentum_5d": 0.02}`` emits a LONG signal when
            momentum_5d > 0.02 and a FLAT signal when < -0.02.
        neutralize: Optional list of factor names to neutralise *before*
            z-scoring (e.g. ``["market_beta"]``). Each target factor is
            residualised against the neutralise factors, stripping out
            the cross-sectional relationship. Applied identically in both
            ``build_cross_section`` (offline calibration) and
            ``_calibrated_signals`` (online inference).
    """

    def __init__(
        self,
        factors: dict[int, dict[str, dict[str, float]]],
        factor_names: Optional[list[str]] = None,
        calibration: "Optional[CalibrationResult]" = None,
        raw_thresholds: Optional[dict[str, float]] = None,
        neutralize: Optional[list[str]] = None,
    ):
        self.factors = factors
        self.factor_names = list(factor_names or FACTOR_NAMES)
        self.calibration = calibration
        self.raw_thresholds = raw_thresholds or {
            # Mild defaults so raw mode produces *some* signals for demo use.
            "momentum_5d": 0.02,
            "rsi_14": 0.30,        # rsi is on [-1,1] scale after rescaling
            "ma_dev_20": 0.02,
        }
        self.neutralize = neutralize

    def _day_factors(self, date: int, bars: dict[str, DailyBar]) -> dict[str, dict[str, float]]:
        """Return the precomputed factors for `date`, restricted to non-suspended symbols."""
        day = self.factors.get(date, {})
        return {s: day[s] for s in bars if not bars[s].is_suspended and s in day}

    def on_cross_section(self, date: int, bars: dict[str, DailyBar]) -> dict[str, list[Signal]]:
        """Process one date's bars and return {symbol: [Signal]}.

        Drops suspended symbols (no factors, no signals).
        """
        factors_by_symbol = self._day_factors(date, bars)
        if not factors_by_symbol:
            return {}

        if self.calibration is not None:
            return self._calibrated_signals(factors_by_symbol)
        return self._raw_signals(factors_by_symbol)

    def build_cross_section(self, date: int, bars: dict[str, DailyBar]) -> CrossSection:
        """Build the standardized factor matrix for one date.

        Used by the calibrator/profiler during the offline phase so they
        share the exact same factor pipeline as the live engine.
        """
        tradeable = {s: b for s, b in bars.items() if not b.is_suspended}
        factors_by_symbol = {s: self.factors.get(date, {}).get(s, {})
                             for s in tradeable if s in self.factors.get(date, {})}
        Z, symbols = standardize_cross_section(
            factors_by_symbol, self.factor_names, neutralize=self.neutralize)
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
        Z, symbols = standardize_cross_section(
            factors_by_symbol, self.factor_names, neutralize=self.neutralize)
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
        """No-op: the engine is stateless (factors are precomputed externally)."""
        pass
