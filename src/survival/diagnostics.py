"""Proportional-hazards diagnostics for fitted Cox models.

Two complementary checks:

1. Scaled Schoenfeld residuals with a slope test (Grambsch & Therneau,
   Biometrika 1994). The Schoenfeld residual for a subject failing at
   ``t_k`` is ``r_k = x_k - xbar(beta_hat, t_k)``, the covariate minus
   the risk-set weighted mean. Under proportional hazards these
   residuals have mean zero at every event time; a systematic trend of
   the scaled residuals ``r_k* = m V r_k`` against a time transform
   ``g(t)`` indicates a time-varying coefficient ``beta(t)``. The score
   test per covariate is

       chi2_j = m * [sum_k w_k (r_k V)_j]^2 / (V_jj * sum_k w_k^2)

   with ``w_k = g(t_k) - gbar``, ``V`` the estimated covariance of
   ``beta_hat`` and ``m`` the number of events; the global version is
   ``m * (w'r) V (r'w) / sum w^2`` on ``p`` degrees of freedom.

2. ``log(-log S(t))`` curves by group. Under proportional hazards
   between groups, ``log(-log S_g(t)) = log HR_g + log(-log S_0(t))``,
   so the curves are vertical shifts of one another (parallel); crossing
   or converging curves indicate a violation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ._validation import (
    FloatArray,
    validate_covariates,
    validate_time_event,
)
from .cox import CoxPHFit, _stratum_indices
from .km import fit_kaplan_meier

Transform = Literal["identity", "log", "rank"]


@dataclass(frozen=True)
class PHTestResult:
    """Result of the proportional-hazards slope test.

    Attributes
    ----------
    table:
        One row per covariate plus a ``GLOBAL`` row, with columns
        ``chi2``, ``df`` and ``p``.
    transform:
        Time transform ``g(t)`` used ("identity", "log" or "rank").
    """

    table: pd.DataFrame
    transform: str

    def violations(self, alpha: float = 0.05) -> list[str]:
        """Covariates whose per-covariate test rejects PH at ``alpha``."""
        rows = self.table.drop(index="GLOBAL")
        return [str(i) for i in rows.index[rows["p"] < alpha]]


def schoenfeld_residuals(
    fit: CoxPHFit,
    X: object,
    time: object,
    event: object,
    *,
    strata: object | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Schoenfeld residuals at the fitted coefficients.

    One residual (a p-vector) per observed event, computed with the same
    tie handling as the fit: with Efron ties, the ``l``-th of ``d`` tied
    events uses the ``l``-adjusted risk-set mean. Residuals are invariant
    to covariate centering.

    Parameters
    ----------
    fit:
        A fitted :class:`~survival.cox.CoxPHFit` (provides ``coef`` and
        the tie method).
    X, time, event, strata:
        The data the model was fitted on.

    Returns
    -------
    tuple
        ``(event_times, residuals)`` sorted by event time ascending;
        ``residuals`` has shape ``(n_events, p)``.
    """
    t, e = validate_time_event(time, event)
    x_arr, _ = validate_covariates(X, fit.names)
    xc = x_arr - x_arr.mean(axis=0)
    strata_idx = _stratum_indices(strata, t.size)
    beta = fit.coef
    efron = fit.ties == "efron"
    p = xc.shape[1]

    res_times: list[float] = []
    res_rows: list[FloatArray] = []
    for idx in strata_idx:
        t_s = t[idx]
        order = np.argsort(-t_s, kind="stable")
        t_o = t_s[order]
        x_o = xc[idx][order]
        e_o = e[idx][order]
        w = np.exp(x_o @ beta)
        m = t_o.size

        s0 = 0.0
        s1 = np.zeros(p)
        i = 0
        while i < m:
            j = i
            while j < m and t_o[j] == t_o[i]:
                j += 1
            block = slice(i, j)
            wb = w[block]
            xb = x_o[block]
            s0 += float(wb.sum())
            s1 += (xb * wb[:, None]).sum(axis=0)

            dmask = e_o[block] == 1
            d = int(dmask.sum())
            if d > 0:
                xd = xb[dmask]
                wd = wb[dmask]
                s0d = float(wd.sum()) if efron else 0.0
                s1d = (
                    (xd * wd[:, None]).sum(axis=0)
                    if efron
                    else np.zeros(p)
                )
                for l in range(d):
                    f = l / d if efron else 0.0
                    mean_l = (s1 - f * s1d) / (s0 - f * s0d)
                    res_times.append(float(t_o[i]))
                    res_rows.append(xd[l] - mean_l)
            i = j

    times_arr = np.asarray(res_times)
    resid = np.vstack(res_rows)
    order = np.argsort(times_arr, kind="stable")
    return times_arr[order], resid[order]


def _transform_times(times: FloatArray, transform: Transform) -> FloatArray:
    """Apply the time transform ``g`` used by the slope test."""
    if transform == "identity":
        return times.astype(float)
    if transform == "log":
        if np.any(times <= 0):
            raise ValueError("log transform requires positive event times")
        return np.log(times)
    if transform == "rank":
        return stats.rankdata(times)
    raise ValueError("transform must be 'identity', 'log' or 'rank'")


def proportional_hazards_test(
    fit: CoxPHFit,
    X: object,
    time: object,
    event: object,
    *,
    strata: object | None = None,
    transform: Transform = "rank",
) -> PHTestResult:
    """Grambsch-Therneau slope test on scaled Schoenfeld residuals.

    A small p-value for a covariate means its residuals trend with
    ``g(t)``: the hazard ratio is not constant over time and the PH
    assumption is violated for that covariate.

    Parameters
    ----------
    fit, X, time, event, strata:
        Fitted model and the data it was fitted on.
    transform:
        Time transform ``g(t)``: ``"rank"`` (default; robust to outlying
        times), ``"identity"`` or ``"log"``.

    Returns
    -------
    PHTestResult
        Per-covariate chi-square tests (1 df each) and a global test
        (p df).
    """
    times, resid = schoenfeld_residuals(
        fit, X, time, event, strata=strata
    )
    m = resid.shape[0]
    g = _transform_times(times, transform)
    w = g - g.mean()
    ssw = float(w @ w)
    v = fit.covariance

    wr = w @ resid  # (p,)
    wrv = wr @ v  # (p,)
    chi2_cov = m * wrv**2 / (np.diag(v) * ssw)
    p_cov = stats.chi2.sf(chi2_cov, 1)
    chi2_global = float(m * (wr @ v @ wr) / ssw)
    p_global = float(stats.chi2.sf(chi2_global, resid.shape[1]))

    table = pd.DataFrame(
        {
            "chi2": np.append(chi2_cov, chi2_global),
            "df": np.append(np.ones(resid.shape[1], dtype=int),
                            resid.shape[1]),
            "p": np.append(p_cov, p_global),
        },
        index=pd.Index(list(fit.names) + ["GLOBAL"], name="covariate"),
    )
    return PHTestResult(table=table, transform=transform)


def log_neg_log_curves(
    time: object, event: object, group: object
) -> dict[object, tuple[FloatArray, FloatArray]]:
    """``(log t, log(-log S(t)))`` per group for a parallelism check.

    Fits Kaplan-Meier within each group and returns the transformed
    curve at event times where ``0 < S < 1`` and ``t > 0``. Parallel
    curves support proportional hazards between the groups.
    """
    t, e = validate_time_event(time, event)
    g = np.asarray(group).ravel()
    if g.size != t.size:
        raise ValueError("group must have the same length as time")
    curves: dict[object, tuple[FloatArray, FloatArray]] = {}
    for lab in np.unique(g):
        mask = g == lab
        km = fit_kaplan_meier(t[mask], e[mask])
        keep = (
            (km.survival > 0.0)
            & (km.survival < 1.0)
            & (km.event_times > 0.0)
        )
        curves[lab] = (
            np.log(km.event_times[keep]),
            np.log(-np.log(km.survival[keep])),
        )
    return curves
