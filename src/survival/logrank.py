"""Log-rank test for comparing survival curves across two or more groups.

At each distinct pooled event time ``t_j`` build the contingency of
events versus at-risk counts. Under the null hypothesis that all groups
share one survival function, the number of events in group ``g`` at
``t_j`` is hypergeometric given the margins, with

    E[d_gj] = d_j * n_gj / n_j
    Cov[d_gj, d_hj] = d_j (n_j - d_j) / (n_j - 1)
                      * (n_gj / n_j) * (delta_gh - n_hj / n_j)

Summing ``O_g - E_g`` and the covariance over event times and dropping
one (redundant) group gives the statistic

    X^2 = z' V^{-1} z,   z = (O - E)_{1..k-1}

which is asymptotically chi-square with ``k - 1`` degrees of freedom.
For ``k = 2`` this reduces to the familiar ``(O_1 - E_1)^2 / V_11``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._validation import FloatArray, validate_time_event


@dataclass(frozen=True)
class LogRankResult:
    """Result of a k-sample log-rank test.

    Attributes
    ----------
    statistic:
        Chi-square statistic.
    df:
        Degrees of freedom (number of groups minus one).
    p_value:
        Asymptotic p-value.
    group_labels:
        Group labels in sorted order; ``observed``/``expected`` follow
        this order.
    observed, expected:
        Total observed and expected event counts per group.
    """

    statistic: float
    df: int
    p_value: float
    group_labels: np.ndarray
    observed: FloatArray
    expected: FloatArray

    def to_frame(self) -> pd.DataFrame:
        """Per-group observed/expected table plus O/E ratio."""
        return pd.DataFrame(
            {
                "group": self.group_labels,
                "observed": self.observed,
                "expected": self.expected,
                "o_over_e": self.observed / self.expected,
            }
        )


def logrank_test(time: object, event: object, group: object) -> LogRankResult:
    """Run the (unweighted) log-rank test across two or more groups.

    Parameters
    ----------
    time:
        Follow-up times, non-negative.
    event:
        1 if the event was observed, 0 if right-censored.
    group:
        Group label per subject (any hashable dtype). At least two
        distinct labels are required.

    Returns
    -------
    LogRankResult
        Chi-square statistic on ``k - 1`` degrees of freedom, p-value,
        and per-group observed/expected event counts.
    """
    t, e = validate_time_event(time, event)
    g = np.asarray(group).ravel()
    if g.size != t.size:
        raise ValueError("group must have the same length as time")
    labels, gidx = np.unique(g, return_inverse=True)
    k = labels.size
    if k < 2:
        raise ValueError("log-rank test requires at least two groups")
    if e.sum() == 0:
        raise ValueError("log-rank test requires at least one event")

    uniq = np.unique(t[e == 1])
    n_times = uniq.size

    # Per-group at-risk and event counts at each pooled event time.
    at_risk = np.empty((k, n_times))
    events = np.empty((k, n_times))
    for gi in range(k):
        tg_sorted = np.sort(t[gidx == gi])
        eg_sorted = np.sort(t[(gidx == gi) & (e == 1)])
        at_risk[gi] = tg_sorted.size - np.searchsorted(
            tg_sorted, uniq, side="left"
        )
        events[gi] = np.searchsorted(
            eg_sorted, uniq, side="right"
        ) - np.searchsorted(eg_sorted, uniq, side="left")

    n_j = at_risk.sum(axis=0)
    d_j = events.sum(axis=0)

    share = at_risk / n_j  # n_gj / n_j
    observed = events.sum(axis=1)
    expected = (share * d_j).sum(axis=1)

    # Hypergeometric variance factor per event time; zero when n_j == 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            n_j > 1, d_j * (n_j - d_j) / (n_j - 1.0), 0.0
        )
    weighted = share * factor  # (k, J)
    cov = np.diag(weighted.sum(axis=1)) - weighted @ share.T

    z_vec = (observed - expected)[: k - 1]
    v_sub = cov[: k - 1, : k - 1]
    try:
        statistic = float(z_vec @ np.linalg.solve(v_sub, z_vec))
    except np.linalg.LinAlgError:
        statistic = float(z_vec @ np.linalg.pinv(v_sub) @ z_vec)
    df = k - 1
    p_value = float(stats.chi2.sf(statistic, df))

    return LogRankResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        group_labels=labels,
        observed=observed,
        expected=expected,
    )
