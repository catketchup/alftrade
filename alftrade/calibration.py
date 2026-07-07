"""
calibration.py - Cross-sectional factor calibration via Fama-MacBeth Ridge
regression.

This is the multi-stock, cross-sectional analogue of hftrade's
``calibration.py``, which was a *time-series* Ridge regression of one
stock's factors against its forward N-tick return. Here we regress the
*cross-section* of stocks' factors against their forward N-day returns,
date by date, and average the resulting β across dates — the canonical
Fama-MacBeth (1973) two-step procedure.

Why Fama-MacBeth instead of one pooled regression
-------------------------------------------------
A pooled regression (stack every (date, stock) pair, run one Ridge) is
simpler and is offered here as a fallback, but it has a well-known flaw:
its standard errors assume the residuals are independent across dates,
which they are not (factor exposures and returns are correlated through
time). Fama-MacBeth fixes this by running a separate cross-sectional
regression per date and reporting

    β_bar  = mean_t(β_t)
    t_stat = mean_t(β_t) / (std_t(β_t) / sqrt(T))

so the time-series variation of β_t itself drives the significance
assessment. This is the standard estimator used in equity factor research.

Two-phase usage (mirrors hftrade)
---------------------------------
    # Phase 1: walk historical bars, collect (date, stock, factors, y) tuples
    calibrator = FactorCalibrator(horizon=5, alpha=1.0)
    for date in sorted(bars_by_date):
        cs = engine.build_cross_section(date, bars_by_date[date])
        # advance return bookkeeping so y pairs with factors from `horizon` days ago
        calibrator.collect(date, cs, bars_by_date)
    result = calibrator.fit()           # Fama-MacBeth by default

    # Phase 2: hand the result to the engine + aggregator
    engine.calibration = result
    aggregator = SignalAggregator(calibration=result)

When calibration is provided:
  - Engine signal strength = Σ β_k · z(factor_k) → "predicted forward return"
  - Aggregator weight       = |IC|                → quality-weighted vote
  - Portfolio long basket   = top-N stocks by predicted return

Stationarity note
-----------------
Unlike hftrade, the daily factors in ``factors.FACTOR_NAMES`` are *already*
stationary (returns, ratios, bounded indicators), so there is no
mid_price-style non-stationary level to remove. The transform applied here
is **cross-sectional standardization** (z-score within each date's
universe), which is a different and equally important concern: it makes β
comparable across factors with different scales (e.g. volatility_20d ~ 0.02
vs rsi_14 ~ 0.5) and removes day-to-day drift in factor dispersions.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json

import numpy as np

from alftrade.data import DailyBar
from alftrade.factors import FACTOR_NAMES
from alftrade.engine import standardize_cross_section


def forward_return(bar_t: DailyBar, bar_tN: DailyBar) -> float:
    """Compute the forward N-day return y_t = (close[t+N] - close[t]) / close[t].

    This is the cross-sectional response variable. It pairs the factors
    observed on day t with the realized return from t to t+N.
    """
    if bar_t.close <= 0:
        return 0.0
    return (bar_tN.close - bar_t.close) / bar_t.close


@dataclass
class CalibrationResult:
    """Output of ``FactorCalibrator.fit``.

    Attributes:
        horizon: N — forward window in trading days used to compute y.
        alpha: Ridge regularization strength used per cross-section.
        method: "fama_macbeth" or "pooled".
        factor_names: Ordered stationary factors calibrated (matches the
            engine's factor_names).
        betas: {factor: β_bar} — averaged cross-sectional coefficients.
            In FM these are the time-series means of per-date β; in pooled
            they are the single Ridge solution. Used by the engine to build
            predicted-return signals.
        fm_t_stats: {factor: t-stat} — FM t-statistic of β_bar (NaN for
            pooled). Measures factor significance: |t| > 2 is a conventional
            threshold.
        fm_p_values: {factor: two-sided p-value} — derived from fm_t_stats
            via the t distribution with T-1 degrees of freedom.
        ics: {factor: pooled Pearson IC} — corr(factor, y) across all
            (date, stock) pairs. Used as aggregation weights.
        ic_mean / ic_std / ir: Per-factor time-series of *per-date*
            cross-sectional IC: mean, std, and IR = mean/std. IR is the
            canonical "Information Ratio" of a factor (consistency of IC).
        r_squared: In FM, the average per-date R²; in pooled, the single R².
        sample_size: Number of (date, stock) pairs used.
        n_dates: Number of cross-sectional regressions run (FM) or dates pooled.
        calibrated_at: ISO timestamp of fit().
    """
    horizon: int
    alpha: float = 1.0
    method: str = "fama_macbeth"
    factor_names: list[str] = field(default_factory=list)
    betas: dict[str, float] = field(default_factory=dict)
    fm_t_stats: dict[str, float] = field(default_factory=dict)
    fm_p_values: dict[str, float] = field(default_factory=dict)
    ics: dict[str, float] = field(default_factory=dict)
    ic_mean: dict[str, float] = field(default_factory=dict)
    ic_std: dict[str, float] = field(default_factory=dict)
    ir: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    sample_size: int = 0
    n_dates: int = 0
    calibrated_at: str = ""

    def __repr__(self) -> str:
        lines = [
            f"CalibrationResult(horizon={self.horizon}, method={self.method}, "
            f"alpha={self.alpha}, n={self.sample_size}, T={self.n_dates}, "
            f"R²={self.r_squared:.4f})"
        ]
        if self.calibrated_at:
            lines[0] += f", date={self.calibrated_at}"
        lines.append(f"  {'Factor':<16} {'Beta':>10} {'t-stat':>8} {'IC':>8} {'IR':>8}")
        lines.append(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for name in self.factor_names:
            b = self.betas.get(name, 0.0)
            t = self.fm_t_stats.get(name, float("nan"))
            ic = self.ics.get(name, 0.0)
            ir = self.ir.get(name, 0.0)
            tstr = f"{t:>8.2f}" if np.isfinite(t) else f"{'--':>8}"
            lines.append(f"  {name:<16} {b:>10.5f} {tstr} {ic:>8.4f} {ir:>8.3f}")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to JSON for the offline → online handoff."""
        data = {
            "horizon": self.horizon,
            "alpha": self.alpha,
            "method": self.method,
            "factor_names": self.factor_names,
            "betas": self.betas,
            "fm_t_stats": self.fm_t_stats,
            "fm_p_values": self.fm_p_values,
            "ics": self.ics,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ir": self.ir,
            "r_squared": self.r_squared,
            "sample_size": self.sample_size,
            "n_dates": self.n_dates,
            "calibrated_at": self.calibrated_at,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        """Restore a CalibrationResult previously saved by save()."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls(
            horizon=d["horizon"],
            alpha=d["alpha"],
            method=d.get("method", "fama_macbeth"),
            factor_names=d["factor_names"],
            betas=d["betas"],
            fm_t_stats=d.get("fm_t_stats", {}),
            fm_p_values=d.get("fm_p_values", {}),
            ics=d["ics"],
            ic_mean=d.get("ic_mean", {}),
            ic_std=d.get("ic_std", {}),
            ir=d.get("ir", {}),
            r_squared=d["r_squared"],
            sample_size=d["sample_size"],
            n_dates=d["n_dates"],
            calibrated_at=d.get("calibrated_at", ""),
        )


class FactorCalibrator:
    """Two-phase cross-sectional factor calibrator.

    Phase 1 — ``collect``: called once per historical date. Stores the
    cross-sectionally standardized factors for every active stock on that
    date, plus the close price needed to later pair with the realized
    forward return. When ``horizon`` trading days have elapsed, the stored
    cross-section is paired with the realized y and the (X, y) sample is
    appended.

    Phase 2 — ``fit``: runs either Fama-MacBeth (default) or pooled Ridge
    regression on the collected samples and returns a ``CalibrationResult``.

    Args:
        horizon: Forward window N (trading days) used to compute y.
        alpha: Ridge L2 strength applied in *each* cross-sectional
            regression (FM) or in the single pooled regression. Larger α
            → more shrinkage → more robust per-date β when the cross-section
            is small or factors are collinear.
        factor_names: Ordered factors to calibrate. Defaults to all 10 in
            ``FACTOR_NAMES``. Must match the engine's factor_names.
        min_samples: Minimum number of (date, stock) pairs for fit() to
            produce a result.
        min_dates: Minimum number of cross-sectional regressions for FM
            t-stats to be meaningful.
    """

    def __init__(
        self,
        horizon: int = 5,
        alpha: float = 1.0,
        factor_names: Optional[list[str]] = None,
        min_samples: int = 100,
        min_dates: int = 10,
    ):
        self.horizon = horizon
        self.alpha = alpha
        self.factor_names = list(factor_names or FACTOR_NAMES)
        self.min_samples = min_samples
        self.min_dates = min_dates

        # Rolling buffer of (date, symbols, z_matrix, closes) waiting for
        # their forward return to become available.
        self._buffer: deque[tuple[int, list[str], np.ndarray, dict[str, float]]] = deque()
        # Collected samples, grouped by date for FM: {date: (z_matrix, y_vector)}.
        self._by_date: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def collect(
        self,
        date: int,
        cross_section,
        bars_by_date: dict[int, dict[str, DailyBar]],
    ) -> None:
        """Phase 1: record one date's standardized factors and pair future y.

        Args:
            date: The trading-day label.
            cross_section: A ``CrossSection`` from ``engine.build_cross_section``
                (its ``z_matrix`` and ``symbols`` are used directly so the
                calibration uses exactly the same standardization as inference).
            bars_by_date: The full panel, so we can look up the close on
                date+N when it arrives.
        """
        Z = cross_section.z_matrix
        symbols = cross_section.symbols
        if len(symbols) == 0:
            return
        closes = {s: cross_section.bars[s].close for s in symbols}
        self._buffer.append((date, symbols, Z, closes))

        # Pair any buffered entries whose horizon has elapsed.
        while self._buffer and self._buffer[0][0] + self.horizon <= date:
            old_date, old_syms, old_Z, old_closes = self._buffer.popleft()
            target_date = old_date + self.horizon
            target_bars = bars_by_date.get(target_date, {})
            # Only keep stocks that have a bar on both dates (survivorship-safe).
            keep_mask = np.array([s in target_bars for s in old_syms])
            if keep_mask.sum() < 2:
                continue
            X = old_Z[keep_mask]
            y = np.array([
                forward_return(
                    # Reconstruct the t-date bar from the close we stored;
                    # forward_return only needs close prices.
                    _CloseOnly(old_closes[s]),
                    target_bars[s],
                )
                for i, s in enumerate(old_syms) if keep_mask[i]
            ])
            self._by_date[old_date] = (X, y)

    @property
    def ready(self) -> bool:
        """True when enough samples and dates have been collected."""
        n = sum(len(y) for _, y in self._by_date.values())
        return n >= self.min_samples and len(self._by_date) >= self.min_dates

    @property
    def num_samples(self) -> int:
        """Total (date, stock) pairs collected so far."""
        return sum(len(y) for _, y in self._by_date.values())

    @property
    def num_dates(self) -> int:
        """Number of cross-sectional regressions available."""
        return len(self._by_date)

    def fit(self, method: str = "fama_macbeth") -> CalibrationResult:
        """Phase 2: run cross-sectional regressions and average.

        Args:
            method: "fama_macbeth" (default) or "pooled".

        Returns:
            CalibrationResult. Raises ValueError if not enough data.
        """
        if self.num_samples < self.min_samples:
            raise ValueError(
                f"Not enough samples: {self.num_samples} < {self.min_samples}."
            )
        if method == "fama_macbeth" and len(self._by_date) < self.min_dates:
            raise ValueError(
                f"Not enough dates for Fama-MacBeth: {len(self._by_date)} < {self.min_dates}."
            )

        from scipy import stats  # local import: scipy is a hard dep but rare-path

        k = len(self.factor_names)
        dates_sorted = sorted(self._by_date.keys())

        if method == "pooled":
            X_all = np.vstack([self._by_date[d][0] for d in dates_sorted])
            y_all = np.concatenate([self._by_date[d][1] for d in dates_sorted])
            betas, r2 = self._ridge(X_all, y_all)
            # Pooled IC and per-factor pooled corr.
            ics, ic_mean, ic_std, ir = self._pooled_ic(X_all, y_all)
            return CalibrationResult(
                horizon=self.horizon,
                alpha=self.alpha,
                method="pooled",
                factor_names=list(self.factor_names),
                betas={n: float(betas[j]) for j, n in enumerate(self.factor_names)},
                fm_t_stats={n: float("nan") for n in self.factor_names},
                fm_p_values={n: float("nan") for n in self.factor_names},
                ics=ics,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ir=ir,
                r_squared=float(r2),
                sample_size=int(len(y_all)),
                n_dates=int(len(dates_sorted)),
                calibrated_at=datetime.now(timezone.utc).isoformat(),
            )

        # ── Fama-MacBeth ──
        beta_series = np.zeros((len(dates_sorted), k))   # β_t per date
        r2_series = np.zeros(len(dates_sorted))
        # Per-date cross-sectional IC (rank IC is more standard for equities;
        # we use Pearson IC for consistency with the engine's signal scale).
        ic_series = np.zeros((len(dates_sorted), k))

        for i, d in enumerate(dates_sorted):
            X, y = self._by_date[d]
            beta_t, r2_t = self._ridge(X, y)
            beta_series[i] = beta_t
            r2_series[i] = r2_t
            for j in range(k):
                col = X[:, j]
                if np.std(col) > 1e-12 and np.std(y) > 1e-12:
                    ic_series[i, j] = float(np.corrcoef(col, y)[0, 1])
                else:
                    ic_series[i, j] = 0.0

        beta_bar = beta_series.mean(axis=0)
        beta_std = beta_series.std(axis=0, ddof=1)
        T = len(dates_sorted)
        # FM t-stat = mean / (std / sqrt(T)). Guard against zero std.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.where(
                beta_std > 1e-12,
                beta_bar / (beta_std / np.sqrt(T)),
                0.0,
            )
        p_values = np.array([
            2 * stats.t.sf(abs(t), df=T - 1) if np.isfinite(t) else 1.0
            for t in t_stats
        ])

        # Pooled IC across all (date, stock) pairs (single number per factor).
        X_all = np.vstack([self._by_date[d][0] for d in dates_sorted])
        y_all = np.concatenate([self._by_date[d][1] for d in dates_sorted])
        ics, ic_mean, ic_std, ir = self._pooled_ic(X_all, y_all, ic_series)

        return CalibrationResult(
            horizon=self.horizon,
            alpha=self.alpha,
            method="fama_macbeth",
            factor_names=list(self.factor_names),
            betas={n: float(beta_bar[j]) for j, n in enumerate(self.factor_names)},
            fm_t_stats={n: float(t_stats[j]) for j, n in enumerate(self.factor_names)},
            fm_p_values={n: float(p_values[j]) for j, n in enumerate(self.factor_names)},
            ics=ics,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            r_squared=float(r2_series.mean()),
            sample_size=int(len(y_all)),
            n_dates=int(T),
            calibrated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _ridge(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Ridge regression on *already standardized* X (no re-standardizing).

        Returns (betas, R²). Since X is cross-sectionally z-scored per date,
        the columns have ~unit variance and the L2 penalty applies fairly.
        """
        k = X.shape[1]
        # Closed-form Ridge: β = (X'X + αI)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        betas = np.linalg.solve(XtX + self.alpha * np.eye(k), Xty)
        y_pred = X @ betas
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return betas, r2

    def _pooled_ic(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        ic_series: Optional[np.ndarray] = None,
    ) -> tuple[dict, dict, dict, dict]:
        """Compute pooled IC plus rolling-IC statistics per factor.

        If ``ic_series`` (per-date IC, shape (T, k)) is provided, IC mean /
        std / IR are taken from the time series (FM-consistent). Otherwise
        they fall back to a single pooled number with IR=0.
        """
        ics, ic_mean, ic_std, ir = {}, {}, {}, {}
        for j, name in enumerate(self.factor_names):
            col = X_all[:, j]
            if np.std(col) > 1e-12 and np.std(y_all) > 1e-12:
                pooled_ic = float(np.corrcoef(col, y_all)[0, 1])
            else:
                pooled_ic = 0.0
            ics[name] = pooled_ic
            if ic_series is not None and ic_series.shape[0] >= 2:
                ts = ic_series[:, j]
                m = float(np.mean(ts))
                s = float(np.std(ts, ddof=1))
                ic_mean[name] = m
                ic_std[name] = s
                ir[name] = float(m / s) if s > 1e-12 else 0.0
            else:
                ic_mean[name] = pooled_ic
                ic_std[name] = 0.0
                ir[name] = 0.0
        return ics, ic_mean, ic_std, ir

    def reset(self) -> None:
        """Clear all collected data."""
        self._buffer.clear()
        self._by_date.clear()


@dataclass
class _CloseOnly:
    """A lightweight stand-in for a DailyBar carrying only close.

    Used when reconstructing the t-date bar inside ``forward_return`` — we
    only need ``.close``, so we avoid re-fetching the full bar from the
    panel.
    """
    close: float
