"""
profiler.py - Cross-sectional factor profiler.

While ``CalibrationResult`` produces the β and IC needed to *build* signals,
``FactorProfiler`` answers a different question: *which factors are actually
good, and how do they behave?* It produces a detailed report with:

  Per-factor cross-sectional quality:
    - Pooled IC and time-series IC (mean / std / IR).
    - Quantile (group) returns: mean forward return per factor quantile,
      averaged across dates. The classic "long-top / short-bottom" spread
      is the single-number summary of a cross-sectional factor's payoff.
    - Monotonicity: Spearman rank correlation between quantile index and
      mean quantile return. High |mono| = the factor sorts stocks correctly.
    - Turnover: how quickly the factor's cross-sectional ranking changes
      day to day (high turnover = expensive to trade).
    - Coverage: fraction of (date, stock) pairs where the factor is
      non-trivially active.

  Per-factor regression quality (mirrors hftrade's profiler):
    - β (FM mean), FM t-stat, FM p-value.
    - Partial R² contribution (drop-one F approximation).

  Model-level quality:
    - Model IC = corr(model prediction, y) pooled.
    - Model monotonicity across quantiles of model prediction.
    - Model quantile returns (does the top quantile of predicted return
      actually realize the highest forward return?).

Usage:
    calibrator = FactorCalibrator(...)
    for date in dates: calibrator.collect(...)
    profiler = FactorProfiler.from_calibrator(calibrator)
    model_profile = profiler.profile()
    print(model_profile)

The profiler shares the calibrator's collected samples so the two are
guaranteed to evaluate the same data.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from alftrade.factors import FACTOR_NAMES


@dataclass
class FactorProfile:
    """Evaluation of a single factor.

    Attributes:
        name: Factor name.
        beta: FM mean β (or pooled β).
        t_stat: FM t-statistic (NaN for pooled).
        p_value: two-sided p-value from the FM t-stat.
        ic: Pooled Pearson IC across all (date, stock) pairs.
        ic_mean: Mean of per-date cross-sectional IC.
        ic_std: Std of per-date cross-sectional IC.
        ir: Information Ratio = ic_mean / ic_std.
        coverage: Fraction of samples where |factor| > epsilon.
        turnover: Mean day-to-day rank turnover of the factor
                  (1 = perfect reshuffle, 0 = frozen ranking).
        quantile_returns: Mean forward y per quantile (lowest → highest
                          factor value), averaged across dates.
        long_short_spread: quantile_returns[-1] - quantile_returns[0].
        monotonicity: Spearman rank corr between quantile index and mean y.
        partial_r2: Drop-one-factor R² contribution (approximate).
    """
    name: str
    beta: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    ic: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    coverage: float = 0.0
    turnover: float = 0.0
    quantile_returns: list[float] = field(default_factory=list)
    long_short_spread: float = 0.0
    monotonicity: float = 0.0
    partial_r2: float = 0.0

    def __repr__(self) -> str:
        sig = ("***" if self.p_value < 0.001 else
               "**"  if self.p_value < 0.01  else
               "*"   if self.p_value < 0.05  else "")
        return (
            f"  {self.name:<16} β={self.beta:>10.5f} t={self.t_stat:>7.2f}{sig:<3} "
            f"IC={self.ic:>7.4f} IR={self.ir:>7.3f} "
            f"L−S={self.long_short_spread:>8.5f} "
            f"turn={self.turnover:>6.3f} mono={self.monotonicity:>7.3f}"
        )


@dataclass
class ModelProfile:
    """Overall model and per-factor evaluation.

    Attributes:
        n_samples: Number of (date, stock) pairs.
        n_dates: Number of cross-sectional dates.
        n_factors: Number of factors.
        model_ic: Pooled corr(model prediction, y).
        model_monotonicity: Spearman rank corr between model-prediction
             quantile index and mean quantile return.
        model_quantile_returns: Mean forward y per model-prediction quantile.
        long_short_spread: Top minus bottom model-quantile mean return.
        factor_profiles: One FactorProfile per factor.
    """
    n_samples: int = 0
    n_dates: int = 0
    n_factors: int = 0
    model_ic: float = 0.0
    model_monotonicity: float = 0.0
    model_quantile_returns: list[float] = field(default_factory=list)
    long_short_spread: float = 0.0
    factor_profiles: list[FactorProfile] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"ModelProfile(n={self.n_samples}, T={self.n_dates}, k={self.n_factors}, "
            f"model_IC={self.model_ic:.4f}, model_mono={self.model_monotonicity:.4f}, "
            f"L−S={self.long_short_spread:.5f})"
        ]
        lines.append(
            f"  {'Factor':<16} {'Beta':>10} {'t-stat':>9}    "
            f"{'IC':>7} {'IR':>7} {'L−S':>9} {'turn':>6} {'mono':>7}"
        )
        lines.append(f"  {'-'*16} {'-'*10} {'-'*9}    {'-'*7} {'-'*7} {'-'*9} {'-'*6} {'-'*7}")
        for fp in self.factor_profiles:
            lines.append(str(fp))
        lines.append("")
        lines.append("  Significance: *** p<0.001  ** p<0.01  * p<0.05")
        return "\n".join(lines)

    def selected_factors(self, p_threshold: float = 0.05) -> list[str]:
        """Return factor names whose FM p-value is below the threshold."""
        return [fp.name for fp in self.factor_profiles if fp.p_value < p_threshold]


class FactorProfiler:
    """Comprehensive cross-sectional factor evaluation.

    Operates on the per-date (z_matrix, y) samples that ``FactorCalibrator``
    collected, so the two never disagree about the underlying data.

    Args:
        by_date: {date: (z_matrix, y_vector)} mapping (the calibrator's
            internal collection).
        factor_names: Ordered factor names matching z_matrix columns.
        alpha: Ridge alpha (used for β / partial-R², consistent with calibrator).
        n_quantiles: Number of quantile buckets (default 5; cross-sections
            are often small so 5 is more robust than 10).
        rolling_window: Window for rolling-IC analysis (in dates).
        coverage_epsilon: |factor| threshold for "active" coverage.
    """

    def __init__(
        self,
        by_date: dict[int, tuple[np.ndarray, np.ndarray]],
        factor_names: Optional[list[str]] = None,
        alpha: float = 1.0,
        n_quantiles: int = 5,
        rolling_window: int = 20,
        coverage_epsilon: float = 1e-8,
    ):
        self.by_date = by_date
        self.factor_names = list(factor_names or FACTOR_NAMES)
        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.rolling_window = rolling_window
        self.coverage_epsilon = coverage_epsilon

    @classmethod
    def from_calibrator(cls, calibrator, **kwargs) -> "FactorProfiler":
        """Build a profiler from a FactorCalibrator's collected samples."""
        return cls(
            by_date=calibrator._by_date,
            factor_names=calibrator.factor_names,
            alpha=calibrator.alpha,
            **kwargs,
        )

    def profile(self) -> ModelProfile:
        """Run all evaluations and return a ModelProfile."""
        dates_sorted = sorted(self.by_date.keys())
        k = len(self.factor_names)
        T = len(dates_sorted)

        # Stack the full panel for pooled metrics.
        X_all = np.vstack([self.by_date[d][0] for d in dates_sorted]) if dates_sorted else np.zeros((0, k))
        y_all = np.concatenate([self.by_date[d][1] for d in dates_sorted]) if dates_sorted else np.zeros(0)
        n = len(y_all)

        # ── 1. Fama-MacBeth β, t-stat, p-value (per factor) ──
        beta_series = np.zeros((T, k)) if T > 0 else np.zeros((0, k))
        r2_full_series = np.zeros(T) if T > 0 else np.zeros(0)
        r2_drop_series = np.zeros((T, k)) if T > 0 else np.zeros((0, k))

        for i, d in enumerate(dates_sorted):
            X, y = self.by_date[d]
            b, r2 = self._ridge(X, y)
            beta_series[i] = b
            r2_full_series[i] = r2
            # Drop-one partial R²: re-fit without factor j, record R² drop.
            for j in range(k):
                mask = [c for c in range(k) if c != j]
                if not mask:
                    continue
                _, r2_dropped = self._ridge(X[:, mask], y)
                r2_drop_series[i, j] = max(r2 - r2_dropped, 0.0)

        if T >= 2:
            beta_bar = beta_series.mean(axis=0)
            beta_std = beta_series.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = np.where(beta_std > 1e-12,
                                   beta_bar / (beta_std / np.sqrt(T)), 0.0)
            p_values = np.array([
                2 * stats.t.sf(abs(t), df=T - 1) if np.isfinite(t) else 1.0
                for t in t_stats
            ])
            partial_r2 = r2_drop_series.mean(axis=0)
        else:
            beta_bar = beta_series.mean(axis=0) if T == 1 else np.zeros(k)
            t_stats = np.zeros(k)
            p_values = np.ones(k)
            partial_r2 = np.zeros(k)

        # ── 2. Pooled IC + per-date IC series ──
        ic_series = np.zeros((T, k)) if T > 0 else np.zeros((0, k))
        for i, d in enumerate(dates_sorted):
            X, y = self.by_date[d]
            for j in range(k):
                col = X[:, j]
                if np.std(col) > 1e-12 and np.std(y) > 1e-12:
                    ic_series[i, j] = float(np.corrcoef(col, y)[0, 1])

        # ── 3. Per-factor quantile returns (cross-sectional, averaged over dates) ──
        # For each date, sort stocks into n_quantiles by factor value, compute
        # mean y per quantile, then average those quantile returns across dates.
        nq = self.n_quantiles
        quantile_acc = np.zeros((T, nq, k)) if T > 0 else np.zeros((0, nq, k))
        for i, d in enumerate(dates_sorted):
            X, y = self.by_date[d]
            if len(y) < nq:
                continue
            for j in range(k):
                order = np.argsort(X[:, j])
                bucket = len(y) // nq
                for q in range(nq):
                    start = q * bucket
                    end = (q + 1) * bucket if q < nq - 1 else len(y)
                    quantile_acc[i, q, j] = float(np.mean(y[order[start:end]]))
        # Average over dates where the bucket was non-trivial.
        quantile_returns_per_factor: list[list[float]] = []
        for j in range(k):
            if T > 0:
                col = quantile_acc[:, :, j]  # (T, nq)
                qr = [float(np.mean(col[:, q])) for q in range(nq)]
            else:
                qr = []
            quantile_returns_per_factor.append(qr)

        # ── 4. Turnover: day-to-day rank change of the factor cross-section ──
        # Mean |Δrank| / (n_stocks − 1), averaged over consecutive date pairs.
        turnover_per_factor = np.zeros(k)
        prev_ranks: Optional[np.ndarray] = None
        rank_diffs = [[] for _ in range(k)]
        for d in dates_sorted:
            X, y = self.by_date[d]
            n_stk = X.shape[0]
            if n_stk < 2:
                prev_ranks = None
                continue
            ranks = np.zeros((n_stk, k))
            for j in range(k):
                ranks[:, j] = stats.rankdata(X[:, j])
            if prev_ranks is not None and prev_ranks.shape[0] == n_stk:
                for j in range(k):
                    # Normalize by max possible rank change.
                    denom = max(n_stk - 1, 1)
                    rank_diffs[j].append(float(np.mean(np.abs(ranks[:, j] - prev_ranks[:, j])) / denom))
            prev_ranks = ranks
        for j in range(k):
            if rank_diffs[j]:
                turnover_per_factor[j] = float(np.mean(rank_diffs[j]))

        # ── 5. Model-level metrics: predicted return = Σ β_bar_j × z_j ──
        if n > 0 and T >= 2:
            y_pred = X_all @ beta_bar
            if np.std(y_pred) > 1e-12 and np.std(y_all) > 1e-12:
                model_ic = float(np.corrcoef(y_pred, y_all)[0, 1])
            else:
                model_ic = 0.0
            model_qr, model_mono = self._quantile_returns(y_pred, y_all)
            model_ls = (model_qr[-1] - model_qr[0]) if model_qr else 0.0
        else:
            model_ic = 0.0
            model_qr, model_mono = [], 0.0
            model_ls = 0.0

        # ── 6. Assemble per-factor profiles ──
        factor_profiles: list[FactorProfile] = []
        for j, name in enumerate(self.factor_names):
            col = X_all[:, j] if n > 0 else np.zeros(0)
            coverage = float(np.mean(np.abs(col) > self.coverage_epsilon)) if n > 0 else 0.0
            pooled_ic = (float(np.corrcoef(col, y_all)[0, 1])
                         if n > 0 and np.std(col) > 1e-12 and np.std(y_all) > 1e-12
                         else 0.0)
            if T >= 2:
                ic_ts = ic_series[:, j]
                ic_m = float(np.mean(ic_ts))
                ic_s = float(np.std(ic_ts, ddof=1))
                ir = float(ic_m / ic_s) if ic_s > 1e-12 else 0.0
            else:
                ic_m, ic_s, ir = pooled_ic, 0.0, 0.0
            qr = quantile_returns_per_factor[j]
            ls = (qr[-1] - qr[0]) if len(qr) >= 2 else 0.0
            if len(qr) >= 3:
                rho, _ = stats.spearmanr(range(len(qr)), qr)
                mono = float(rho) if np.isfinite(rho) else 0.0
            else:
                mono = 0.0
            factor_profiles.append(FactorProfile(
                name=name,
                beta=float(beta_bar[j]),
                t_stat=float(t_stats[j]),
                p_value=float(p_values[j]),
                ic=pooled_ic,
                ic_mean=ic_m,
                ic_std=ic_s,
                ir=ir,
                coverage=coverage,
                turnover=float(turnover_per_factor[j]),
                quantile_returns=qr,
                long_short_spread=float(ls),
                monotonicity=mono,
                partial_r2=float(partial_r2[j]) if T > 0 else 0.0,
            ))

        return ModelProfile(
            n_samples=int(n),
            n_dates=int(T),
            n_factors=int(k),
            model_ic=model_ic,
            model_monotonicity=float(model_mono),
            model_quantile_returns=[float(x) for x in model_qr],
            long_short_spread=float(model_ls),
            factor_profiles=factor_profiles,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _ridge(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        k = X.shape[1]
        XtX = X.T @ X
        Xty = X.T @ y
        betas = np.linalg.solve(XtX + self.alpha * np.eye(k), Xty)
        y_pred = X @ betas
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return betas, r2

    def _quantile_returns(self, pred: np.ndarray, y: np.ndarray) -> tuple[list[float], float]:
        """Sort by prediction, bucket into n_quantiles, return mean y + monotonicity."""
        nq = self.n_quantiles
        n = len(pred)
        if n < nq:
            return [], 0.0
        order = np.argsort(pred)
        bucket = n // nq
        qr = []
        for q in range(nq):
            start = q * bucket
            end = (q + 1) * bucket if q < nq - 1 else n
            qr.append(float(np.mean(y[order[start:end]])))
        if len(qr) >= 3:
            rho, _ = stats.spearmanr(range(len(qr)), qr)
            mono = float(rho) if np.isfinite(rho) else 0.0
        else:
            mono = 0.0
        return qr, mono
