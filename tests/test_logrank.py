"""Log-rank test: hand-computed lifetables and published results."""
import numpy as np
import pytest

from survival import logrank_test
from survival.simulate import simulate_two_groups

from test_km import FREIREICH_EVENT, FREIREICH_TIME

# Gehan (1965) placebo arm: all 21 relapsed.
PLACEBO_TIME = [
    1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23,
]


def test_logrank_hand_computed_two_by_two_lifetables():
    # Group A: events at 1 and 3. Group B: event at 2, censored at 4.
    # t=1: n=4, d=1 in A -> E_A += 2/4,  V += 1*3/3 * (1/2)(1/2) = 1/4
    # t=2: n=3, d=1 in B -> E_A += 1/3,  V += 1*2/2 * (1/3)(2/3) = 2/9
    # t=3: n=2, d=1 in A -> E_A += 1/2,  V += 1*1/1 * (1/2)(1/2) = 1/4
    # O_A = 2, E_A = 4/3, V = 13/18, chi2 = (2/3)^2 / (13/18) = 8/13.
    res = logrank_test(
        [1, 3, 2, 4], [1, 1, 1, 0], ["A", "A", "B", "B"]
    )
    assert res.statistic == pytest.approx(8 / 13)
    assert res.df == 1
    assert res.expected[0] == pytest.approx(4 / 3)
    assert res.observed[0] == 2


def test_logrank_matches_published_gehan_result():
    # The classic 6-MP vs placebo comparison; the log-rank chi-square is
    # 16.79 with p ~= 4.2e-5 (e.g. R survival::survdiff on the gehan
    # data, and Kleinbaum & Klein ch. 2).
    time = np.array(FREIREICH_TIME + PLACEBO_TIME, dtype=float)
    event = np.array(FREIREICH_EVENT + [1] * 21)
    group = np.array(["6-MP"] * 21 + ["placebo"] * 21)
    res = logrank_test(time, event, group)
    assert res.statistic == pytest.approx(16.793, abs=5e-3)
    assert res.p_value == pytest.approx(4.17e-5, rel=5e-3)
    # Expected counts as published: E(6-MP) ~= 19.25, E(placebo) ~= 10.75
    assert res.expected[0] == pytest.approx(19.25, abs=0.01)
    assert res.expected[1] == pytest.approx(10.75, abs=0.01)


def test_logrank_matches_published_aml_result():
    # R survival 'aml' data: survdiff reports chi-square 3.4, p = 0.07.
    time = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
            5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    event = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
             1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    group = [0] * 11 + [1] * 12
    res = logrank_test(time, event, group)
    assert res.statistic == pytest.approx(3.396, abs=5e-3)
    assert res.p_value == pytest.approx(0.065, abs=2e-3)


def test_logrank_identical_groups_gives_zero_statistic():
    time = [3, 5, 7, 9, 11, 3, 5, 7, 9, 11]
    event = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    group = [0] * 5 + [1] * 5
    res = logrank_test(time, event, group)
    assert res.statistic == pytest.approx(0.0, abs=1e-12)
    assert res.p_value == pytest.approx(1.0)


def test_logrank_three_groups_has_two_degrees_of_freedom():
    rng = np.random.default_rng(3)
    time = rng.exponential(10, 90)
    event = np.ones(90, dtype=int)
    group = np.repeat(["a", "b", "c"], 30)
    res = logrank_test(time, event, group)
    assert res.df == 2
    assert 0.0 <= res.p_value <= 1.0
    assert res.observed.sum() == pytest.approx(res.expected.sum())


def test_logrank_type_one_error_close_to_nominal():
    # 200 seeded null replicates; the 0.05-level rejection rate has
    # Monte-Carlo sd ~ 0.015, so [0.01, 0.10] is a generous check.
    rng = np.random.default_rng(41)
    rejections = 0
    for _ in range(200):
        time, event, group = simulate_two_groups(
            120, hazard_ratio=1.0, seed=rng
        )
        rejections += logrank_test(time, event, group).p_value < 0.05
    assert 0.01 <= rejections / 200 <= 0.10


def test_logrank_input_validation():
    with pytest.raises(ValueError):
        logrank_test([1, 2, 3], [1, 1, 0], ["a", "a", "a"])
    with pytest.raises(ValueError):
        logrank_test([1, 2], [0, 0], ["a", "b"])
    with pytest.raises(ValueError):
        logrank_test([1, 2], [1, 1], ["a"])
