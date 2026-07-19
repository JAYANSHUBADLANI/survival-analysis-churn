"""Synthetic right-censored churn cohorts with known ground truth.

Event times follow a Weibull proportional-hazards model:

    h(t | x) = (shape / scale) * (t / scale)^(shape - 1) * exp(x' beta)
    S(t | x) = exp( -(t / scale)^shape * exp(x' beta) )

sampled by inversion: with ``U ~ Uniform(0, 1)``,

    T = scale * ( -log(U) / exp(x' beta) )^(1 / shape).

Censoring is independent ``C ~ Uniform(0, censor_max)`` (an enrollment /
snapshot mechanism); the observed pair is ``(min(T, C), 1{T <= C})``.
The default settings yield roughly one third censored, matching the
"about 30-40% censored" regime typical of subscription churn data.

The default cohort has four covariates with known effects:

    month_to_month  ~ Bernoulli(0.55)      beta = +0.90  (HR 2.46)
    auto_pay        ~ Bernoulli(0.40)      beta = -0.50  (HR 0.61)
    support_calls   ~ std. Poisson(1.5)    beta = +0.35  (HR 1.42)
    monthly_spend   ~ Normal(0, 1)         beta = +0.25  (HR 1.28)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._validation import FloatArray

DEFAULT_SHAPE = 1.3
DEFAULT_SCALE = 24.0
DEFAULT_CENSOR_MAX = 60.0
DEFAULT_BETAS: dict[str, float] = {
    "month_to_month": 0.90,
    "auto_pay": -0.50,
    "support_calls": 0.35,
    "monthly_spend": 0.25,
}


def true_survival(
    t: object,
    *,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
    linpred: float = 0.0,
) -> FloatArray:
    """True survival ``S(t | x)`` of the Weibull PH model.

    ``linpred`` is the linear predictor ``x' beta`` (0 for baseline).
    """
    t_arr = np.asarray(t, dtype=float)
    return np.exp(-((t_arr / scale) ** shape) * np.exp(linpred))


def weibull_time_for_survival(
    s: float,
    *,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
) -> float:
    """Time at which the baseline survival equals ``s`` (inverse of S)."""
    if not 0.0 < s < 1.0:
        raise ValueError("s must be in (0, 1)")
    return float(scale * (-np.log(s)) ** (1.0 / shape))


def _observe(
    rng: np.random.Generator,
    event_time: FloatArray,
    censor_max: float,
) -> tuple[FloatArray, np.ndarray]:
    """Apply independent uniform censoring on ``(0, censor_max)``."""
    c = rng.uniform(0.0, censor_max, size=event_time.size)
    observed = np.minimum(event_time, c)
    event = (event_time <= c).astype(int)
    return observed, event


def draw_event_times(
    rng: np.random.Generator,
    linpred: FloatArray,
    *,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
) -> FloatArray:
    """Draw Weibull PH event times by inverse-transform sampling."""
    u = rng.uniform(size=linpred.size)
    return scale * (-np.log(u) / np.exp(linpred)) ** (1.0 / shape)


def simulate_weibull_cohort(
    n: int,
    *,
    seed: int | np.random.Generator,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
    censor_max: float = DEFAULT_CENSOR_MAX,
) -> tuple[FloatArray, np.ndarray]:
    """Homogeneous (no-covariate) censored Weibull cohort.

    Returns ``(time, event)``; the estimand is the known baseline
    ``S(t) = exp(-(t / scale)^shape)``.
    """
    rng = np.random.default_rng(seed)
    t_event = draw_event_times(rng, np.zeros(n), shape=shape, scale=scale)
    return _observe(rng, t_event, censor_max)


def simulate_two_groups(
    n_per_group: int,
    *,
    hazard_ratio: float,
    seed: int | np.random.Generator,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
    censor_max: float = DEFAULT_CENSOR_MAX,
) -> tuple[FloatArray, np.ndarray, np.ndarray]:
    """Two censored groups whose hazards differ by a known ratio.

    Group 1's hazard is ``hazard_ratio`` times group 0's at every time
    (proportional hazards holds exactly). Returns
    ``(time, event, group)``.
    """
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n_per_group)
    linpred = np.log(hazard_ratio) * group
    t_event = draw_event_times(rng, linpred, shape=shape, scale=scale)
    time, event = _observe(rng, t_event, censor_max)
    return time, event, group


def simulate_churn_cohort(
    n: int,
    *,
    seed: int | np.random.Generator,
    shape: float = DEFAULT_SHAPE,
    scale: float = DEFAULT_SCALE,
    betas: dict[str, float] | None = None,
    censor_max: float = DEFAULT_CENSOR_MAX,
) -> pd.DataFrame:
    """Simulate a right-censored churn cohort with known hazard ratios.

    Parameters
    ----------
    n:
        Cohort size.
    seed:
        Integer seed or a ``numpy`` Generator.
    shape, scale:
        Weibull baseline hazard parameters (months).
    betas:
        Log hazard ratios keyed by covariate name; defaults to
        :data:`DEFAULT_BETAS`.
    censor_max:
        Upper bound of the uniform censoring distribution.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, ``event`` and one column per covariate
        (in ``betas`` key order).
    """
    rng = np.random.default_rng(seed)
    b = dict(DEFAULT_BETAS if betas is None else betas)
    covs: dict[str, FloatArray] = {}
    for name in b:
        if name == "month_to_month":
            covs[name] = rng.binomial(1, 0.55, n).astype(float)
        elif name == "auto_pay":
            covs[name] = rng.binomial(1, 0.40, n).astype(float)
        elif name == "support_calls":
            raw = rng.poisson(1.5, n).astype(float)
            covs[name] = (raw - 1.5) / np.sqrt(1.5)
        else:
            covs[name] = rng.standard_normal(n)

    linpred = np.zeros(n)
    for name, beta in b.items():
        linpred += beta * covs[name]
    t_event = draw_event_times(rng, linpred, shape=shape, scale=scale)
    time, event = _observe(rng, t_event, censor_max)

    out = {"time": time, "event": event}
    out.update(covs)
    return pd.DataFrame(out)
