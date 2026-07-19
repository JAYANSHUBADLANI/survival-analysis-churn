"""Cox proportional hazards regression fit by Newton-Raphson.

The model is ``h(t | x) = h_0(t) exp(x' beta)`` with ``h_0`` left
unspecified. ``beta`` maximizes the log partial likelihood; with Efron's
approximation for the ``d_j`` events tied at time ``t_j`` (tie set
``D_j``, risk set ``R_j``):

    l(beta) = sum_j [ sum_{i in D_j} x_i' beta
              - sum_{l=0}^{d_j - 1} log( S0(R_j) - (l / d_j) s0(D_j) ) ]

where ``S0(R) = sum_{i in R} exp(x_i' beta)`` and ``s0(D)`` is the same
sum over the tie set. Breslow's simpler approximation sets ``l/d_j = 0``.
Efron is the default here: it is exact when ties arise from coarse
measurement of continuous times and is markedly less biased than
Breslow when ties are heavy (e.g. tenure recorded in whole months).

The gradient and Hessian are accumulated analytically alongside the
log likelihood using the risk-set sums ``S0`` (scalar), ``S1`` (p,) and
``S2`` (p, p); each Efron step ``l`` contributes mean ``m1`` and
curvature ``m2 - m1 m1'`` computed from the adjusted sums. Standard
errors come from the inverse of the observed information at the
optimum. Optional stratification fits a shared ``beta`` while giving
every stratum its own baseline hazard (the partial likelihood is the
sum of within-stratum partial likelihoods).

Covariates are centered internally before optimization. The partial
likelihood depends on covariates only through differences ``x_i - x_l``
within risk sets, so centering changes nothing statistically but keeps
``exp(x' beta)`` well conditioned.
"""
from __future__ import annotations

import warnings
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

Ties = Literal["efron", "breslow"]


@dataclass(frozen=True)
class CoxPHFit:
    """Result of a Cox proportional hazards fit.

    Attributes
    ----------
    names:
        Covariate names, in the column order of the design matrix.
    coef:
        Estimated log hazard ratios ``beta_hat``.
    se:
        Standard errors from the inverse observed information.
    hazard_ratio, hr_ci_lower, hr_ci_upper:
        ``exp(beta_hat)`` with Wald ``1 - alpha`` confidence limits.
    z, p_values:
        Wald statistics ``beta_hat / se`` and two-sided p-values.
    covariance:
        Full estimated covariance matrix of ``beta_hat``.
    loglik, loglik_null:
        Log partial likelihood at the optimum and at ``beta = 0``.
    lr_statistic, lr_df, lr_p_value:
        Likelihood-ratio test of the global null ``beta = 0``.
    n, n_events:
        Sample size and number of observed events.
    ties:
        Tie-handling method used ("efron" or "breslow").
    n_strata:
        Number of strata (1 when unstratified).
    alpha:
        Significance level for the confidence limits.
    iterations:
        Newton-Raphson iterations performed.
    converged:
        Whether the convergence criterion was met.
    """

    names: list[str]
    coef: FloatArray
    se: FloatArray
    hazard_ratio: FloatArray
    hr_ci_lower: FloatArray
    hr_ci_upper: FloatArray
    z: FloatArray
    p_values: FloatArray
    covariance: FloatArray
    loglik: float
    loglik_null: float
    lr_statistic: float
    lr_df: int
    lr_p_value: float
    n: int
    n_events: int
    ties: str
    n_strata: int
    alpha: float
    iterations: int
    converged: bool

    def summary_frame(self) -> pd.DataFrame:
        """Coefficient table (one row per covariate)."""
        ci_pct = 100 * (1 - self.alpha)
        return pd.DataFrame(
            {
                "coef": self.coef,
                "hazard_ratio": self.hazard_ratio,
                "se(coef)": self.se,
                "z": self.z,
                "p": self.p_values,
                f"hr_lower_{ci_pct:g}": self.hr_ci_lower,
                f"hr_upper_{ci_pct:g}": self.hr_ci_upper,
            },
            index=pd.Index(self.names, name="covariate"),
        )


def _stratum_indices(
    strata: object | None, n: int
) -> list[FloatArray]:
    """Row indices per stratum (a single stratum when ``strata is None``)."""
    if strata is None:
        return [np.arange(n)]
    s = np.asarray(strata).ravel()
    if s.size != n:
        raise ValueError("strata must have the same length as time")
    return [np.flatnonzero(s == lab) for lab in np.unique(s)]


def _loglik_no_ties(
    beta: FloatArray,
    t_o: FloatArray,
    x_o: FloatArray,
    e_o: np.ndarray,
) -> tuple[float, FloatArray, FloatArray]:
    """Vectorized contribution for one stratum with all-distinct times.

    With no ties, Efron and Breslow coincide and every risk-set sum is a
    reverse cumulative sum over subjects sorted by increasing time,
    which evaluates the whole stratum in O(n p^2) vectorized work.
    Expects inputs sorted by increasing time.
    """
    eta = x_o @ beta
    w = np.exp(eta)
    s0 = np.cumsum(w[::-1])[::-1]
    s1 = np.cumsum((x_o * w[:, None])[::-1], axis=0)[::-1]
    outer = np.einsum("ij,ik->ijk", x_o, x_o) * w[:, None, None]
    s2 = np.cumsum(outer[::-1], axis=0)[::-1]

    ev = e_o == 1
    s0e = s0[ev]
    m1 = s1[ev] / s0e[:, None]
    ll = float(eta[ev].sum() - np.log(s0e).sum())
    grad = x_o[ev].sum(axis=0) - m1.sum(axis=0)
    hess = -(
        (s2[ev] / s0e[:, None, None]).sum(axis=0)
        - np.einsum("ij,ik->jk", m1, m1)
    )
    return ll, grad, hess


def _loglik_with_ties(
    beta: FloatArray,
    t_o: FloatArray,
    x_o: FloatArray,
    e_o: np.ndarray,
    ties: Ties,
) -> tuple[float, FloatArray, FloatArray]:
    """Loop contribution for one stratum, handling tied event times.

    Iterates subjects in decreasing time order so the risk-set sums
    ``S0``, ``S1``, ``S2`` accumulate incrementally, then applies the
    Efron (or Breslow) correction within each tie set. Expects inputs
    sorted by decreasing time.
    """
    p = x_o.shape[1]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    eta = x_o @ beta
    w = np.exp(eta)
    m = t_o.size

    s0 = 0.0
    s1 = np.zeros(p)
    s2 = np.zeros((p, p))
    i = 0
    while i < m:
        j = i
        while j < m and t_o[j] == t_o[i]:
            j += 1
        block = slice(i, j)
        wb = w[block]
        xb = x_o[block]
        xw = xb * wb[:, None]
        s0 += float(wb.sum())
        s1 += xw.sum(axis=0)
        s2 += xw.T @ xb

        dmask = e_o[block] == 1
        d = int(dmask.sum())
        if d > 0:
            xd = xb[dmask]
            wd = wb[dmask]
            ll += float(eta[block][dmask].sum())
            grad += xd.sum(axis=0)
            if ties == "efron":
                s0d = float(wd.sum())
                xdw = xd * wd[:, None]
                s1d = xdw.sum(axis=0)
                s2d = xdw.T @ xd
            else:  # breslow
                s0d = 0.0
                s1d = np.zeros(p)
                s2d = np.zeros((p, p))
            for l in range(d):
                f = l / d if ties == "efron" else 0.0
                z0 = s0 - f * s0d
                m1 = (s1 - f * s1d) / z0
                m2 = (s2 - f * s2d) / z0
                ll -= np.log(z0)
                grad -= m1
                hess -= m2 - np.outer(m1, m1)
        i = j

    return ll, grad, hess


def partial_loglik(
    beta: FloatArray,
    X: FloatArray,
    time: FloatArray,
    event: np.ndarray,
    strata_idx: list[FloatArray],
    ties: Ties,
) -> tuple[float, FloatArray, FloatArray]:
    """Log partial likelihood with analytic gradient and Hessian.

    Dispatches per stratum: a fully vectorized path when all times in
    the stratum are distinct (ties play no role and Efron equals
    Breslow), and an incremental tie-set loop otherwise.

    Returns
    -------
    tuple
        ``(loglik, gradient, hessian)``; the Hessian is that of the log
        likelihood (negative definite near the optimum).
    """
    p = X.shape[1]
    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))

    for idx in strata_idx:
        t_s = time[idx]
        if np.unique(t_s).size == t_s.size:
            order = np.argsort(t_s, kind="stable")
            part = _loglik_no_ties(
                beta, t_s[order], X[idx][order], event[idx][order]
            )
        else:
            order = np.argsort(-t_s, kind="stable")
            part = _loglik_with_ties(
                beta, t_s[order], X[idx][order], event[idx][order], ties
            )
        ll += part[0]
        grad += part[1]
        hess += part[2]

    return ll, grad, hess


def fit_cox_ph(
    X: object,
    time: object,
    event: object,
    *,
    names: list[str] | None = None,
    ties: Ties = "efron",
    strata: object | None = None,
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> CoxPHFit:
    """Fit a Cox proportional hazards model by Newton-Raphson.

    Parameters
    ----------
    X:
        Covariate matrix ``(n, p)`` (DataFrame columns become names).
    time:
        Follow-up times, non-negative.
    event:
        1 if the event was observed, 0 if right-censored.
    names:
        Covariate names; overrides DataFrame columns if given.
    ties:
        Tie handling: ``"efron"`` (default, recommended) or
        ``"breslow"``. Identical when all event times are distinct.
    strata:
        Optional stratum label per subject. Each stratum keeps its own
        baseline hazard; ``beta`` is shared.
    alpha:
        Significance level for Wald confidence limits.
    max_iter, tol:
        Newton-Raphson iteration cap and relative log-likelihood
        convergence tolerance (with step-halving on non-increase),
        matching the criterion used by standard Cox implementations.

    Returns
    -------
    CoxPHFit

    Raises
    ------
    ValueError
        On invalid input, zero events, or a singular information matrix
        (e.g. collinear covariates).
    """
    t, e = validate_time_event(time, event)
    x_arr, resolved = validate_covariates(X, names)
    if x_arr.shape[0] != t.size:
        raise ValueError("X must have one row per subject")
    if ties not in ("efron", "breslow"):
        raise ValueError("ties must be 'efron' or 'breslow'")
    if e.sum() == 0:
        raise ValueError("Cox model requires at least one event")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    xc = x_arr - x_arr.mean(axis=0)
    strata_idx = _stratum_indices(strata, t.size)
    p = xc.shape[1]

    beta = np.zeros(p)
    ll, grad, hess = partial_loglik(beta, xc, t, e, strata_idx, ties)
    loglik_null = ll

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        try:
            delta = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError as err:
            raise ValueError(
                "singular information matrix; check for collinear or "
                "constant covariates"
            ) from err
        step = 1.0
        for _ in range(40):
            beta_new = beta + step * delta
            ll_new, grad_new, hess_new = partial_loglik(
                beta_new, xc, t, e, strata_idx, ties
            )
            if np.isfinite(ll_new) and ll_new >= ll - 1e-12:
                break
            step *= 0.5
        else:
            break
        improvement = ll_new - ll
        beta, ll, grad, hess = beta_new, ll_new, grad_new, hess_new
        if abs(improvement) <= tol * (abs(ll) + tol):
            converged = True
            break

    if not converged:
        warnings.warn(
            "Newton-Raphson did not meet the convergence criterion; "
            "estimates may be unreliable (possible monotone likelihood).",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        covariance = np.linalg.inv(-hess)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "singular information matrix at the optimum; check for "
            "collinear or constant covariates"
        ) from err

    se = np.sqrt(np.diag(covariance))
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    z_stat = beta / se
    p_values = 2.0 * stats.norm.sf(np.abs(z_stat))
    lr_statistic = 2.0 * (ll - loglik_null)
    lr_p = float(stats.chi2.sf(lr_statistic, p))

    return CoxPHFit(
        names=resolved,
        coef=beta,
        se=se,
        hazard_ratio=np.exp(beta),
        hr_ci_lower=np.exp(beta - z_crit * se),
        hr_ci_upper=np.exp(beta + z_crit * se),
        z=z_stat,
        p_values=p_values,
        covariance=covariance,
        loglik=float(ll),
        loglik_null=float(loglik_null),
        lr_statistic=float(lr_statistic),
        lr_df=int(p),
        lr_p_value=lr_p,
        n=int(t.size),
        n_events=int(e.sum()),
        ties=ties,
        n_strata=len(strata_idx),
        alpha=float(alpha),
        iterations=iterations,
        converged=converged,
    )
