"""Survival analysis for customer churn, implemented from scratch.

Public API:

- :func:`fit_kaplan_meier` / :class:`KaplanMeierFit`
- :func:`logrank_test` / :class:`LogRankResult`
- :func:`fit_cox_ph` / :class:`CoxPHFit`
- :func:`proportional_hazards_test`, :func:`schoenfeld_residuals`,
  :func:`log_neg_log_curves`
- Simulators with known ground truth in :mod:`survival.simulate`
"""
from .cox import CoxPHFit, fit_cox_ph
from .diagnostics import (
    PHTestResult,
    log_neg_log_curves,
    proportional_hazards_test,
    schoenfeld_residuals,
)
from .km import KaplanMeierFit, fit_kaplan_meier
from .logrank import LogRankResult, logrank_test
from .simulate import (
    simulate_churn_cohort,
    simulate_two_groups,
    simulate_weibull_cohort,
    true_survival,
)

__version__ = "1.0.0"

__all__ = [
    "CoxPHFit",
    "KaplanMeierFit",
    "LogRankResult",
    "PHTestResult",
    "fit_cox_ph",
    "fit_kaplan_meier",
    "log_neg_log_curves",
    "logrank_test",
    "proportional_hazards_test",
    "schoenfeld_residuals",
    "simulate_churn_cohort",
    "simulate_two_groups",
    "simulate_weibull_cohort",
    "true_survival",
]
