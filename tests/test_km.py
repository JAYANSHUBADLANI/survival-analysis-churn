"""Kaplan-Meier estimator: textbook values, Greenwood, CI behavior."""
import numpy as np
import pytest

from survival import fit_kaplan_meier

# Freireich et al. (1963) 6-MP arm, the classic remission-duration data
# reproduced in Kleinbaum & Klein, "Survival Analysis". n = 21, 9 events.
FREIREICH_TIME = [
    6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32,
    32, 34, 35,
]
FREIREICH_EVENT = [
    1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
]
# Published estimates: S(6)=.857, S(7)=.807, S(10)=.753, S(13)=.690,
# S(16)=.627, S(22)=.538, S(23)=.448.
FREIREICH_SURVIVAL = [
    0.857143, 0.806723, 0.752941, 0.690196, 0.627451, 0.537815, 0.448179,
]


def test_km_matches_published_freireich_values():
    fit = fit_kaplan_meier(FREIREICH_TIME, FREIREICH_EVENT)
    np.testing.assert_allclose(
        fit.event_times, [6, 7, 10, 13, 16, 22, 23]
    )
    np.testing.assert_allclose(
        fit.survival, FREIREICH_SURVIVAL, atol=1e-6
    )
    np.testing.assert_array_equal(fit.at_risk, [21, 17, 15, 12, 11, 7, 6])
    np.testing.assert_array_equal(fit.events, [3, 1, 1, 1, 1, 1, 1])


def test_greenwood_se_matches_published_freireich_values():
    # Kleinbaum & Klein report se(S(6)) = 0.0764 and se(S(7)) = 0.0869.
    fit = fit_kaplan_meier(FREIREICH_TIME, FREIREICH_EVENT)
    se = np.sqrt(fit.variance)
    assert se[0] == pytest.approx(0.0764, abs=5e-4)
    assert se[1] == pytest.approx(0.0869, abs=5e-4)


def test_greenwood_variance_hand_computed():
    # times 1, 2, 4, all events, n = 3:
    # S(1) = 2/3, Var = (2/3)^2 * [1/(3*2)]           = 4/54
    # S(2) = 1/3, Var = (1/3)^2 * [1/(3*2) + 1/(2*1)] = 2/27
    # S(4) = 0  -> Greenwood variance undefined (nan).
    fit = fit_kaplan_meier([1, 2, 4], [1, 1, 1])
    np.testing.assert_allclose(fit.survival, [2 / 3, 1 / 3, 0.0])
    assert fit.variance[0] == pytest.approx(4 / 54)
    assert fit.variance[1] == pytest.approx(2 / 27)
    assert np.isnan(fit.variance[2])


def test_km_no_censoring_matches_empirical_survival():
    rng = np.random.default_rng(0)
    t = rng.exponential(5.0, 400)
    fit = fit_kaplan_meier(t, np.ones_like(t))
    # With no censoring, KM is the empirical survival function.
    empirical = 1.0 - np.arange(1, 401) / 400.0
    np.testing.assert_allclose(
        fit.survival, empirical[np.searchsorted(np.sort(t),
                                                fit.event_times)],
        atol=1e-12,
    )
    assert fit.survival[-1] == pytest.approx(0.0)


def test_km_censored_tied_with_event_counted_at_risk():
    # Standard convention: a subject censored at t is at risk at t.
    fit = fit_kaplan_meier([2, 2], [1, 0])
    np.testing.assert_allclose(fit.survival, [0.5])
    np.testing.assert_array_equal(fit.at_risk, [2])


def test_km_confidence_limits_bracket_estimate_within_unit_interval():
    fit = fit_kaplan_meier(FREIREICH_TIME, FREIREICH_EVENT)
    assert np.all(fit.ci_lower > 0.0)
    assert np.all(fit.ci_upper < 1.0)
    assert np.all(fit.ci_lower < fit.survival)
    assert np.all(fit.survival < fit.ci_upper)


def test_km_step_function_evaluation():
    fit = fit_kaplan_meier([5, 10], [1, 1])
    values = fit.survival_at([0.0, 4.999, 5.0, 7.5, 10.0, 99.0])
    np.testing.assert_allclose(values, [1.0, 1.0, 0.5, 0.5, 0.0, 0.0])


def test_km_median_and_all_censored():
    fit = fit_kaplan_meier([5, 10], [1, 1])
    assert fit.median_survival_time() == 5.0
    flat = fit_kaplan_meier([3, 6, 9], [0, 0, 0])
    assert flat.event_times.size == 0
    assert flat.survival_at([1.0, 100.0]) == pytest.approx([1.0, 1.0])
    assert np.isnan(flat.median_survival_time())


def test_km_input_validation():
    with pytest.raises(ValueError):
        fit_kaplan_meier([1, 2], [1])
    with pytest.raises(ValueError):
        fit_kaplan_meier([-1, 2], [1, 1])
    with pytest.raises(ValueError):
        fit_kaplan_meier([1, 2], [1, 2])
    with pytest.raises(ValueError):
        fit_kaplan_meier([], [])
