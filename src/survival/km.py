"""Kaplan-Meier estimation of the survival function under right censoring.

The Kaplan-Meier estimator is the nonparametric maximum-likelihood
estimator of ``S(t) = P(T > t)`` from right-censored data. With distinct
event times ``t_1 < t_2 < ...``, ``d_i`` events at ``t_i``, and ``n_i``
subjects at risk just before ``t_i``:

    S_hat(t) = prod_{i : t_i <= t} (1 - d_i / n_i)

Variance is estimated with Greenwood's formula,

    Var[S_hat(t)] = S_hat(t)^2 * sum_{i : t_i <= t} d_i / (n_i (n_i - d_i)),

and pointwise confidence intervals are built on the complementary
log-log scale, ``theta = log(-log S)``, then back-transformed as
``S^exp(+/- z * se_theta)``. This keeps the interval inside [0, 1] and
behaves far better near the boundaries than the naive linear interval.

Conventions: subjects censored at time ``t`` are counted as at risk at
``t`` (events precede censorings at tied times), the standard convention
for right-censored data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._validation import FloatArray, IntArray, validate_time_event


@dataclass(frozen=True)
class KaplanMeierFit:
    """Result of a Kaplan-Meier fit.

    All arrays are indexed by the distinct observed event times, in
    increasing order. The estimated curve is a right-continuous step
    function: ``S(t) = survival[i]`` for ``event_times[i] <= t <
    event_times[i+1]``, and ``S(t) = 1`` before the first event time.

    Attributes
    ----------
    event_times:
        Distinct times at which at least one event occurred.
    survival:
        ``S_hat`` evaluated at each event time.
    variance:
        Greenwood variance of ``S_hat``. ``nan`` where the estimate
        reaches 0 (the variance formula degenerates there).
    ci_lower, ci_upper:
        Pointwise ``1 - alpha`` confidence limits (log-log transform).
    at_risk:
        Number at risk just before each event time.
    events:
        Number of events at each event time.
    n_subjects, n_events:
        Sample size and total number of observed events.
    alpha:
        Significance level used for the confidence limits.
    """

    event_times: FloatArray
    survival: FloatArray
    variance: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    at_risk: IntArray
    events: IntArray
    n_subjects: int
    n_events: int
    alpha: float

    def survival_at(self, times: object) -> FloatArray:
        """Evaluate the step-function estimate ``S_hat`` at arbitrary times.

        Returns 1.0 for times before the first observed event.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        step_values = np.concatenate(([1.0], self.survival))
        idx = np.searchsorted(self.event_times, t, side="right")
        return step_values[idx]

    def ci_at(self, times: object) -> tuple[FloatArray, FloatArray]:
        """Evaluate the confidence limits at arbitrary times.

        Returns ``(nan, nan)`` for times before the first observed event,
        where no uncertainty statement is available.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        lo = np.concatenate(([np.nan], self.ci_lower))
        hi = np.concatenate(([np.nan], self.ci_upper))
        idx = np.searchsorted(self.event_times, t, side="right")
        return lo[idx], hi[idx]

    def median_survival_time(self) -> float:
        """Smallest time with ``S_hat(t) <= 0.5``; ``nan`` if never reached."""
        below = self.survival <= 0.5
        if not below.any():
            return float("nan")
        return float(self.event_times[int(np.argmax(below))])

    def to_frame(self) -> pd.DataFrame:
        """Lifetable as a DataFrame (one row per distinct event time)."""
        return pd.DataFrame(
            {
                "time": self.event_times,
                "at_risk": self.at_risk,
                "events": self.events,
                "survival": self.survival,
                "std_err": np.sqrt(self.variance),
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            }
        )


def fit_kaplan_meier(
    time: object, event: object, *, alpha: float = 0.05
) -> KaplanMeierFit:
    """Fit the Kaplan-Meier estimator to right-censored data.

    Parameters
    ----------
    time:
        Follow-up times (event or censoring), non-negative.
    event:
        1 if the event was observed, 0 if right-censored.
    alpha:
        Significance level for the pointwise confidence limits
        (default 0.05 for 95% intervals).

    Returns
    -------
    KaplanMeierFit
        Estimated curve, Greenwood variance, and log-log confidence
        limits at each distinct event time. If no events occurred the
        estimate is the constant 1 (all arrays empty).
    """
    t, e = validate_time_event(time, event)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = t.size
    t_sorted = np.sort(t)
    event_times_all = np.sort(t[e == 1])
    uniq = np.unique(event_times_all)

    # Risk set just before each event time: everyone with time >= t_i.
    at_risk = n - np.searchsorted(t_sorted, uniq, side="left")
    d = (
        np.searchsorted(event_times_all, uniq, side="right")
        - np.searchsorted(event_times_all, uniq, side="left")
    )

    survival = np.cumprod(1.0 - d / at_risk)

    # Greenwood's formula. Where the risk set is exhausted (n_i == d_i,
    # so S_hat drops to 0) the summand is infinite and the variance is
    # undefined; report nan there and beyond.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = d / (at_risk * (at_risk - d).astype(float))
        terms = np.where(at_risk > d, terms, np.nan)
        variance = survival**2 * np.cumsum(terms)

        z = stats.norm.ppf(1.0 - alpha / 2.0)
        se_cll = np.sqrt(variance) / np.abs(survival * np.log(survival))
        ci_lower = survival ** np.exp(z * se_cll)
        ci_upper = survival ** np.exp(-z * se_cll)

    undefined = ~np.isfinite(se_cll)
    ci_lower = np.where(undefined, np.nan, ci_lower)
    ci_upper = np.where(undefined, np.nan, ci_upper)

    return KaplanMeierFit(
        event_times=uniq,
        survival=survival,
        variance=variance,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        at_risk=at_risk.astype(int),
        events=d.astype(int),
        n_subjects=int(n),
        n_events=int(e.sum()),
        alpha=float(alpha),
    )
