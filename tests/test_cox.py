"""Cox model: analytic MLE, published values, derivative checks."""
import numpy as np
import pytest

from survival import fit_cox_ph
from survival.cox import (
    _loglik_no_ties,
    _loglik_with_ties,
    _stratum_indices,
    partial_loglik,
)
from survival.simulate import simulate_churn_cohort

# R survival 'aml' (leukemia maintenance) data; x = 1 for Nonmaintained.
AML_TIME = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
            5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
AML_EVENT = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
             1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
AML_X = [0.0] * 11 + [1.0] * 12

# Small fixed dataset with tied event times; reference coefficients and
# standard errors verified against statsmodels PHReg (which matches R
# survival::coxph) for both tie-handling methods.
TIED_TIME = [3, 3, 5, 5, 5, 7, 8, 8, 10, 12, 14, 15]
TIED_EVENT = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]
TIED_X = np.column_stack(
    [
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0.5, 1.2, -0.3, 0.7, 1.5, -1.0, 0.2, 0.8, -0.6, 1.1, 0.4, -0.2],
    ]
).astype(float)


def test_cox_analytic_three_subject_mle():
    # Subjects (t, event, x): (1,1,1), (2,1,0), (3,1,1). The partial
    # likelihood is e^b/(2e^b+1) * 1/(e^b+1), maximized where
    # 2u^2 = 1 with u = e^b, i.e. beta = -log(2)/2. The information is
    # m1(1-m1) summed over the first two risk sets (binary covariate).
    fit = fit_cox_ph(
        np.array([1.0, 0.0, 1.0]), [1, 2, 3], [1, 1, 1]
    )
    beta_expected = -0.5 * np.log(2.0)
    assert fit.coef[0] == pytest.approx(beta_expected, abs=1e-8)
    u = np.exp(beta_expected)
    ll_expected = np.log(u / (2 * u + 1)) + np.log(1 / (u + 1))
    assert fit.loglik == pytest.approx(ll_expected, abs=1e-10)
    m1a, m1b = 2 * u / (2 * u + 1), u / (u + 1)
    info = m1a * (1 - m1a) + m1b * (1 - m1b)
    assert fit.se[0] == pytest.approx(info**-0.5, abs=1e-8)
    assert fit.converged


def test_cox_matches_published_aml_efron():
    # R survival::coxph(Surv(time, status) ~ x, aml) with Efron ties:
    # coef = 0.9155, se = 0.5119 for x = Nonmaintained.
    fit = fit_cox_ph(np.array(AML_X), AML_TIME, AML_EVENT)
    assert fit.coef[0] == pytest.approx(0.9155, abs=1e-4)
    assert fit.se[0] == pytest.approx(0.5119, abs=1e-4)
    assert fit.hazard_ratio[0] == pytest.approx(2.498, abs=2e-3)


def test_cox_matches_reference_on_tied_data_both_methods():
    fit_e = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT, ties="efron")
    np.testing.assert_allclose(
        fit_e.coef, [0.81025858, 0.29219337], atol=1e-6
    )
    np.testing.assert_allclose(
        fit_e.se, [0.82199464, 0.61136492], atol=1e-6
    )
    assert fit_e.loglik == pytest.approx(-14.69196216, abs=1e-6)

    fit_b = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT, ties="breslow")
    np.testing.assert_allclose(
        fit_b.coef, [0.79975598, 0.26856514], atol=1e-6
    )
    np.testing.assert_allclose(
        fit_b.se, [0.82550408, 0.61185857], atol=1e-6
    )


def test_cox_efron_equals_breslow_without_ties():
    rng = np.random.default_rng(5)
    n = 60
    x = rng.standard_normal((n, 2))
    t = rng.exponential(np.exp(-0.5 * x[:, 0]))
    e = rng.uniform(size=n) < 0.8
    fit_e = fit_cox_ph(x, t, e.astype(int), ties="efron")
    fit_b = fit_cox_ph(x, t, e.astype(int), ties="breslow")
    np.testing.assert_allclose(fit_e.coef, fit_b.coef, atol=1e-10)
    np.testing.assert_allclose(fit_e.se, fit_b.se, atol=1e-10)


def test_cox_efron_differs_from_breslow_with_ties():
    fit_e = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT, ties="efron")
    fit_b = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT, ties="breslow")
    assert np.max(np.abs(fit_e.coef - fit_b.coef)) > 1e-3


def test_cox_wald_quantities_are_consistent():
    fit = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT)
    np.testing.assert_allclose(fit.hazard_ratio, np.exp(fit.coef))
    np.testing.assert_allclose(fit.z, fit.coef / fit.se)
    from scipy import stats

    np.testing.assert_allclose(
        fit.p_values, 2 * stats.norm.sf(np.abs(fit.z))
    )
    z975 = stats.norm.ppf(0.975)
    np.testing.assert_allclose(
        fit.hr_ci_lower, np.exp(fit.coef - z975 * fit.se)
    )
    np.testing.assert_allclose(
        fit.hr_ci_upper, np.exp(fit.coef + z975 * fit.se)
    )
    assert fit.lr_statistic == pytest.approx(
        2 * (fit.loglik - fit.loglik_null)
    )
    assert fit.lr_statistic >= 0.0


def test_cox_vectorized_path_matches_tie_loop_exactly():
    # With all-distinct times the reverse-cumsum fast path and the
    # incremental tie-set loop compute the same quantities.
    rng = np.random.default_rng(11)
    n, p = 120, 3
    x = rng.standard_normal((n, p))
    t = rng.exponential(5, n)  # continuous: distinct w.p. 1
    e = (rng.uniform(size=n) < 0.7).astype(int)
    beta = np.array([0.5, -0.4, 0.2])
    asc = np.argsort(t)
    fast = _loglik_no_ties(beta, t[asc], x[asc], e[asc])
    desc = np.argsort(-t)
    slow = _loglik_with_ties(beta, t[desc], x[desc], e[desc], "efron")
    assert fast[0] == pytest.approx(slow[0], abs=1e-10)
    np.testing.assert_allclose(fast[1], slow[1], atol=1e-10)
    np.testing.assert_allclose(fast[2], slow[2], atol=1e-10)


def test_cox_gradient_and_hessian_match_finite_differences():
    rng = np.random.default_rng(9)
    n, p = 80, 3
    x = rng.standard_normal((n, p))
    t = np.ceil(rng.exponential(5, n))  # heavy ties
    e = (rng.uniform(size=n) < 0.7).astype(int)
    idx = _stratum_indices(None, n)
    beta = np.array([0.4, -0.3, 0.2])
    ll, grad, hess = partial_loglik(beta, x, t, e, idx, "efron")
    eps = 1e-6
    for j in range(p):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += eps
        bm[j] -= eps
        lp, gp, _ = partial_loglik(bp, x, t, e, idx, "efron")
        lm, gm, _ = partial_loglik(bm, x, t, e, idx, "efron")
        assert grad[j] == pytest.approx((lp - lm) / (2 * eps), abs=1e-4)
        np.testing.assert_allclose(
            hess[:, j], (gp - gm) / (2 * eps), atol=1e-4
        )


def test_cox_recovers_simulation_truth():
    names = ["month_to_month", "auto_pay", "support_calls", "monthly_spend"]
    truth = np.array([0.90, -0.50, 0.35, 0.25])
    df = simulate_churn_cohort(1500, seed=21)
    fit = fit_cox_ph(df[names], df["time"], df["event"])
    assert fit.converged
    assert np.max(np.abs(fit.coef - truth)) < 0.15
    # 95% Wald intervals should bracket the truth for this seed.
    lo = fit.coef - 1.96 * fit.se
    hi = fit.coef + 1.96 * fit.se
    assert np.all((lo <= truth) & (truth <= hi))


def test_cox_stratified_single_stratum_matches_pooled():
    fit = fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT)
    strat = fit_cox_ph(
        TIED_X, TIED_TIME, TIED_EVENT, strata=np.zeros(len(TIED_TIME))
    )
    np.testing.assert_allclose(fit.coef, strat.coef, atol=1e-12)
    assert strat.n_strata == 1


def test_cox_stratified_two_strata_runs_and_pools_information():
    rng = np.random.default_rng(13)
    n = 300
    x = rng.standard_normal((n, 2))
    stratum = (rng.uniform(size=n) < 0.5).astype(int)
    base = np.where(stratum == 1, 4.0, 10.0)
    # Exponential scale = base * exp(-0.6 x) means the hazard is
    # (1/base) * exp(+0.6 x): the true log hazard ratio is +0.6.
    t = rng.exponential(base * np.exp(-0.6 * x[:, 0]))
    e = (rng.uniform(size=n) < 0.8).astype(int)
    fit = fit_cox_ph(x, t, e, strata=stratum)
    assert fit.n_strata == 2
    assert fit.converged
    assert np.all(np.isfinite(fit.se))
    assert fit.coef[0] == pytest.approx(0.6, abs=0.25)


def test_cox_singular_design_raises():
    x = np.column_stack([TIED_X[:, 0], TIED_X[:, 0]])  # collinear
    with pytest.raises(ValueError, match="singular|collinear"):
        fit_cox_ph(x, TIED_TIME, TIED_EVENT)


def test_cox_input_validation():
    with pytest.raises(ValueError):
        fit_cox_ph(TIED_X, TIED_TIME, [0] * 12)  # no events
    with pytest.raises(ValueError):
        fit_cox_ph(TIED_X, TIED_TIME, TIED_EVENT, ties="exact")
    with pytest.raises(ValueError):
        fit_cox_ph(TIED_X[:5], TIED_TIME, TIED_EVENT)
