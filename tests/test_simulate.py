"""Simulator: distributional correctness and end-to-end recovery."""
import numpy as np
import pytest
from scipy import stats

from survival import fit_cox_ph, fit_kaplan_meier
from survival.simulate import (
    DEFAULT_SCALE,
    DEFAULT_SHAPE,
    draw_event_times,
    simulate_churn_cohort,
    simulate_two_groups,
    simulate_weibull_cohort,
    true_survival,
    weibull_time_for_survival,
)


def test_event_times_follow_the_target_weibull():
    rng = np.random.default_rng(2)
    t = draw_event_times(rng, np.zeros(3000))
    ks = stats.kstest(
        t, stats.weibull_min(DEFAULT_SHAPE, scale=DEFAULT_SCALE).cdf
    )
    assert ks.pvalue > 0.01


def test_covariates_scale_the_hazard_as_specified():
    # Under h(t|x) = h0(t) e^(b x), a fitted Cox model on a large
    # two-group sample must recover b = log(hazard_ratio).
    time, event, group = simulate_two_groups(
        3000, hazard_ratio=2.0, seed=6
    )
    fit = fit_cox_ph(group.astype(float), time, event, names=["group"])
    assert fit.coef[0] == pytest.approx(np.log(2.0), abs=0.1)


def test_censoring_fraction_in_target_band():
    _, e1 = simulate_weibull_cohort(20000, seed=3)
    df = simulate_churn_cohort(20000, seed=4)
    _, e3, _ = simulate_two_groups(10000, hazard_ratio=1.0, seed=5)
    for frac in (
        1 - e1.mean(),
        1 - df["event"].mean(),
        1 - e3.mean(),
    ):
        assert 0.25 <= frac <= 0.45


def test_true_survival_closed_form_values():
    # At t = scale the exponent is exactly exp(linpred).
    assert true_survival(DEFAULT_SCALE) == pytest.approx(np.exp(-1))
    assert true_survival(
        DEFAULT_SCALE, linpred=np.log(2)
    ) == pytest.approx(np.exp(-2))
    assert true_survival(0.0) == pytest.approx(1.0)
    t_half = weibull_time_for_survival(0.5)
    assert true_survival(t_half) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        weibull_time_for_survival(1.5)


def test_simulation_is_reproducible_and_seed_sensitive():
    a = simulate_churn_cohort(200, seed=42)
    b = simulate_churn_cohort(200, seed=42)
    c = simulate_churn_cohort(200, seed=43)
    assert a.equals(b)
    assert not a.equals(c)
    assert list(a.columns) == [
        "time", "event", "month_to_month", "auto_pay",
        "support_calls", "monthly_spend",
    ]


def test_km_recovers_true_curve_on_large_cohort():
    time, event = simulate_weibull_cohort(5000, seed=31)
    fit = fit_kaplan_meier(time, event)
    max_dev = np.max(
        np.abs(fit.survival - true_survival(fit.event_times))
    )
    assert max_dev < 0.04


def test_greenwood_interval_covers_truth_at_median():
    # 200 seeded replicates at the true median: empirical coverage of
    # the 95% interval should sit near 0.95 (MC sd ~ 0.015).
    t_med = weibull_time_for_survival(0.5)
    rng = np.random.default_rng(51)
    covered = 0
    for _ in range(200):
        time, event = simulate_weibull_cohort(200, seed=rng)
        fit = fit_kaplan_meier(time, event)
        lo, hi = fit.ci_at(np.array([t_med]))
        covered += bool(lo[0] <= 0.5 <= hi[0])
    assert 0.90 <= covered / 200 <= 0.995
