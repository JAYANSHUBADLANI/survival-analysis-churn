"""PH diagnostics: residual properties and violation detection."""
import numpy as np
import pytest

from survival import (
    fit_cox_ph,
    log_neg_log_curves,
    proportional_hazards_test,
    schoenfeld_residuals,
)
from survival.simulate import simulate_churn_cohort

NAMES = ["month_to_month", "auto_pay", "support_calls", "monthly_spend"]


def _fitted_cohort(seed: int, n: int):
    df = simulate_churn_cohort(n, seed=seed)
    fit = fit_cox_ph(df[NAMES], df["time"], df["event"])
    return df, fit


def test_schoenfeld_residuals_sum_to_zero_at_mle():
    # The summed Schoenfeld residuals equal the score at beta_hat,
    # which is zero at the maximum of the partial likelihood.
    df, fit = _fitted_cohort(seed=17, n=400)
    times, resid = schoenfeld_residuals(
        fit, df[NAMES], df["time"], df["event"]
    )
    np.testing.assert_allclose(resid.sum(axis=0), 0.0, atol=1e-5)
    assert resid.shape == (fit.n_events, len(NAMES))
    assert np.all(np.diff(times) >= 0)


def test_ph_test_detects_nonproportional_hazards():
    # Two groups with different Weibull shapes (one decreasing hazard,
    # one increasing): the hazard ratio crosses 1 over time, a textbook
    # PH violation the slope test must flag decisively.
    rng = np.random.default_rng(1)
    n = 400
    group = np.repeat([0.0, 1.0], n)
    t0 = 15 * rng.weibull(0.7, n)
    t1 = 15 * rng.weibull(2.2, n)
    t = np.concatenate([t0, t1])
    c = rng.uniform(0, 40, 2 * n)
    obs = np.minimum(t, c)
    ev = (t <= c).astype(int)
    fit = fit_cox_ph(group[:, None], obs, ev, names=["group"])
    res = proportional_hazards_test(fit, group[:, None], obs, ev)
    assert res.table.loc["group", "p"] < 1e-10
    assert "group" in res.violations()


def test_ph_test_clean_under_proportional_hazards():
    # The simulator satisfies PH exactly, so no covariate should be
    # flagged at any stringent level (seeded; per-covariate p >= 0.01).
    df, fit = _fitted_cohort(seed=10, n=800)
    res = proportional_hazards_test(
        fit, df[NAMES], df["time"], df["event"]
    )
    per_covariate = res.table.drop(index="GLOBAL")["p"]
    assert (per_covariate > 0.01).all()
    assert res.violations(alpha=0.01) == []


def test_ph_test_transforms_and_validation():
    df, fit = _fitted_cohort(seed=19, n=300)
    for transform in ("identity", "log", "rank"):
        res = proportional_hazards_test(
            fit, df[NAMES], df["time"], df["event"], transform=transform
        )
        assert set(res.table.columns) == {"chi2", "df", "p"}
        assert np.all(res.table["chi2"] >= 0)
    with pytest.raises(ValueError):
        proportional_hazards_test(
            fit, df[NAMES], df["time"], df["event"], transform="sqrt"
        )


def test_log_neg_log_curves_shapes_and_finiteness():
    rng = np.random.default_rng(23)
    t = rng.exponential([6, 12], size=(150, 2)).T.ravel()
    e = (rng.uniform(size=300) < 0.8).astype(int)
    g = np.repeat(["fast", "slow"], 150)
    curves = log_neg_log_curves(t, e, g)
    assert set(curves) == {"fast", "slow"}
    for log_t, cll in curves.values():
        assert log_t.shape == cll.shape
        assert np.all(np.isfinite(log_t))
        assert np.all(np.isfinite(cll))
        assert np.all(np.diff(cll) >= 0)  # -log S is nondecreasing


def test_proportional_group_curves_are_roughly_parallel():
    # Exponential hazards with a constant ratio: the two log(-log S)
    # curves should differ by ~log(HR) in the well-estimated region.
    rng = np.random.default_rng(29)
    n = 4000
    t = np.concatenate(
        [rng.exponential(10, n), rng.exponential(5, n)]
    )
    e = np.ones(2 * n, dtype=int)
    g = np.repeat([0, 1], n)
    curves = log_neg_log_curves(t, e, g)
    x0, y0 = curves[0]
    x1, y1 = curves[1]
    grid = np.linspace(0.5, 1.5, 12)  # log-time window in both supports
    gap = np.interp(grid, x1, y1) - np.interp(grid, x0, y0)
    np.testing.assert_allclose(gap, np.log(2), atol=0.12)
