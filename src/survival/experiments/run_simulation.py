"""Layer-1 validation: every estimator against simulated known truth.

Four studies, all seeded and fully reproducible:

1. **Kaplan-Meier recovery** - one large homogeneous Weibull cohort;
   overlay the KM step function on the true ``S(t)`` and report the
   maximum absolute deviation at event times.
2. **Greenwood coverage** - many small cohorts; empirical coverage of
   the pointwise 95% log-log intervals at fixed times, compared with
   the nominal level and its Monte-Carlo error.
3. **Log-rank calibration** - type-I error under identical hazards and
   power under known hazard ratios.
4. **Cox recovery** - a single large fit (estimates vs truth with CIs)
   plus repeated fits reporting bias, empirical SD vs mean model SE,
   and 95% CI coverage per covariate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cox import fit_cox_ph
from ..km import fit_kaplan_meier
from ..logrank import logrank_test
from ..plotting import forest_plot, plot_km_fits
from ..simulate import (
    DEFAULT_BETAS,
    DEFAULT_CENSOR_MAX,
    DEFAULT_SCALE,
    DEFAULT_SHAPE,
    simulate_churn_cohort,
    simulate_two_groups,
    simulate_weibull_cohort,
    true_survival,
    weibull_time_for_survival,
)
from ._output import ensure_dir, save_figure, write_json, write_table

COVERAGE_SURVIVAL_LEVELS = (0.9, 0.75, 0.5, 0.3)
POWER_HAZARD_RATIOS = (1.3, 1.6)


def km_recovery(outdir: Path, seed: int, n: int) -> dict:
    """Overlay KM on the true baseline survival; report max |error|."""
    time, event = simulate_weibull_cohort(n, seed=seed)
    fit = fit_kaplan_meier(time, event)
    truth = true_survival(fit.event_times)
    max_abs_dev = float(np.max(np.abs(fit.survival - truth)))

    grid = np.linspace(0.0, float(time.max()), 300)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_km_fits(
        ax,
        {f"Kaplan-Meier (n={n})": fit},
        true_curve=(grid, true_survival(grid)),
        xlabel="months",
        title=(
            "Kaplan-Meier recovery of the true Weibull survival\n"
            f"max |KM - truth| at event times = {max_abs_dev:.4f}"
        ),
    )
    save_figure(fig, outdir / "km_recovery.png")

    frame = fit.to_frame()
    frame["true_survival"] = truth
    frame["abs_error"] = np.abs(frame["survival"] - truth)
    write_table(
        frame.iloc[:: max(1, len(frame) // 40)],
        outdir,
        "km_recovery",
        index=False,
    )
    return {
        "n": n,
        "n_events": fit.n_events,
        "censored_fraction": 1.0 - fit.n_events / n,
        "max_abs_deviation": max_abs_dev,
    }


def greenwood_coverage(
    outdir: Path, seed: int, reps: int, n: int
) -> dict:
    """Empirical pointwise coverage of the 95% log-log intervals."""
    eval_times = np.array(
        [weibull_time_for_survival(s) for s in COVERAGE_SURVIVAL_LEVELS]
    )
    truth = true_survival(eval_times)
    covered = np.zeros(eval_times.size)
    informative = np.zeros(eval_times.size)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        time, event = simulate_weibull_cohort(n, seed=rng)
        fit = fit_kaplan_meier(time, event)
        lo, hi = fit.ci_at(eval_times)
        ok = np.isfinite(lo) & np.isfinite(hi)
        informative += ok
        covered += ok & (lo <= truth) & (truth <= hi)
    coverage = covered / informative
    mc_err = 1.96 * np.sqrt(0.95 * 0.05 / reps)

    table = pd.DataFrame(
        {
            "eval_time": eval_times,
            "true_survival": truth,
            "empirical_coverage": coverage,
            "reps": informative.astype(int),
        }
    )
    write_table(table, outdir, "greenwood_coverage", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0.95, color="black", ls="--", lw=1.2, label="nominal 0.95")
    ax.axhspan(
        0.95 - mc_err,
        0.95 + mc_err,
        color="grey",
        alpha=0.25,
        label=f"Monte-Carlo band ({reps} reps)",
    )
    ax.plot(eval_times, coverage, "o-", color="#1f77b4", label="empirical")
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("evaluation time (months)")
    ax.set_ylabel("coverage of 95% CI")
    ax.set_title("Greenwood log-log interval coverage "
                 f"(n={n} per replicate)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, outdir / "greenwood_coverage.png")

    return {
        "reps": reps,
        "n": n,
        "coverage_by_time": {
            f"S={lvl}": float(cov)
            for lvl, cov in zip(COVERAGE_SURVIVAL_LEVELS, coverage)
        },
        "mc_error_95": float(mc_err),
    }


def logrank_calibration(
    outdir: Path, seed: int, reps_null: int, reps_power: int, n_per_group: int
) -> dict:
    """Type-I error under H0 and power under known hazard ratios.

    The null is evaluated at two sample sizes: the asymptotic chi-square
    reference is known to be slightly liberal in moderate samples, so
    showing the rejection rate approach 0.05 as n grows is part of the
    calibration evidence.
    """
    rng = np.random.default_rng(seed)
    null_rows = []
    null_p_small = np.empty(reps_null)
    for size in (n_per_group, 400):
        rejections = 0
        for r in range(reps_null):
            time, event, group = simulate_two_groups(
                size, hazard_ratio=1.0, seed=rng
            )
            p = logrank_test(time, event, group).p_value
            rejections += p < 0.05
            if size == n_per_group:
                null_p_small[r] = p
        null_rows.append(
            {
                "hazard_ratio": 1.0,
                "n_per_group": size,
                "rejection_rate_at_0.05": rejections / reps_null,
                "reps": reps_null,
            }
        )
    type1 = float(null_rows[0]["rejection_rate_at_0.05"])
    type1_large = float(null_rows[1]["rejection_rate_at_0.05"])

    power_rows = []
    for hr in POWER_HAZARD_RATIOS:
        rej = 0
        for _ in range(reps_power):
            time, event, group = simulate_two_groups(
                n_per_group, hazard_ratio=hr, seed=rng
            )
            rej += logrank_test(time, event, group).p_value < 0.05
        power_rows.append(
            {
                "hazard_ratio": hr,
                "n_per_group": n_per_group,
                "rejection_rate_at_0.05": rej / reps_power,
                "reps": reps_power,
            }
        )
    table = pd.DataFrame(null_rows + power_rows)
    write_table(table, outdir, "logrank_calibration", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(
        null_p_small,
        bins=20,
        range=(0, 1),
        density=True,
        color="#1f77b4",
        alpha=0.75,
        edgecolor="white",
    )
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Uniform(0,1)")
    ax.set_xlabel("log-rank p-value under H0")
    ax.set_ylabel("density")
    ax.set_title(
        f"Null p-values ({reps_null} reps, n={n_per_group}/group); "
        f"type-I error at 0.05 = {type1:.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, outdir / "logrank_calibration.png")

    return {
        "n_per_group": n_per_group,
        "type1_error": type1,
        "type1_error_n400": type1_large,
        "reps_null": reps_null,
        "power": {
            str(row["hazard_ratio"]): float(
                row["rejection_rate_at_0.05"]
            )
            for row in power_rows
        },
    }


def cox_recovery(
    outdir: Path, seed: int, n_single: int, reps: int, n_rep: int
) -> dict:
    """Single-fit truth table plus repeated-simulation calibration."""
    names = list(DEFAULT_BETAS)
    true_beta = np.array([DEFAULT_BETAS[k] for k in names])

    cohort = simulate_churn_cohort(n_single, seed=seed)
    fit = fit_cox_ph(cohort[names], cohort["time"], cohort["event"])
    single = fit.summary_frame()
    single.insert(0, "true_hr", np.exp(true_beta))
    lower_col = next(c for c in single.columns if c.startswith("hr_lower"))
    upper_col = next(c for c in single.columns if c.startswith("hr_upper"))
    single["ci_contains_truth"] = (single[lower_col] <= single["true_hr"]) & (
        single["true_hr"] <= single[upper_col]
    )
    write_table(single, outdir, "cox_recovery_single")

    fig, ax = plt.subplots(figsize=(7, 4))
    forest_plot(
        ax,
        single,
        truth={k: float(np.exp(DEFAULT_BETAS[k])) for k in names},
        title=f"Cox hazard-ratio recovery (single fit, n={n_single})",
    )
    save_figure(fig, outdir / "cox_recovery.png")

    rng = np.random.default_rng(seed + 1)
    estimates = np.empty((reps, len(names)))
    ses = np.empty((reps, len(names)))
    covered = np.zeros(len(names))
    for r in range(reps):
        rep = simulate_churn_cohort(n_rep, seed=rng)
        f = fit_cox_ph(rep[names], rep["time"], rep["event"])
        estimates[r] = f.coef
        ses[r] = f.se
        covered += (f.coef - 1.959964 * f.se <= true_beta) & (
            true_beta <= f.coef + 1.959964 * f.se
        )
    repeated = pd.DataFrame(
        {
            "true_beta": true_beta,
            "mean_estimate": estimates.mean(axis=0),
            "bias": estimates.mean(axis=0) - true_beta,
            "empirical_sd": estimates.std(axis=0, ddof=1),
            "mean_model_se": ses.mean(axis=0),
            "ci95_coverage": covered / reps,
        },
        index=pd.Index(names, name="covariate"),
    )
    write_table(repeated, outdir, "cox_recovery_repeated")

    return {
        "n_single": n_single,
        "single_fit": {
            name: {
                "true_hr": float(np.exp(tb)),
                "hr": float(hr),
                "ci": [float(lo), float(hi)],
            }
            for name, tb, hr, lo, hi in zip(
                names,
                true_beta,
                single["hazard_ratio"],
                single[lower_col],
                single[upper_col],
            )
        },
        "repeated": {
            "reps": reps,
            "n_per_rep": n_rep,
            "max_abs_bias": float(repeated["bias"].abs().max()),
            "coverage": {
                name: float(c)
                for name, c in zip(names, repeated["ci95_coverage"])
            },
        },
        "censored_fraction_single": 1.0
        - float(cohort["event"].mean()),
    }


def main(
    *, outdir: str = "results", seed: int = 2026, fast: bool = False
) -> dict:
    """Run all four studies and write figures/tables plus summary files."""
    out = ensure_dir(Path(outdir) / "simulation")
    scale = 10 if fast else 1
    summary = {
        "seed": seed,
        "baseline": {
            "distribution": "Weibull",
            "shape": DEFAULT_SHAPE,
            "scale": DEFAULT_SCALE,
            "censoring": f"Uniform(0, {DEFAULT_CENSOR_MAX})",
        },
        "km_recovery": km_recovery(out, seed, n=2000),
        "greenwood_coverage": greenwood_coverage(
            out, seed + 10, reps=500 // scale, n=300
        ),
        "logrank_calibration": logrank_calibration(
            out,
            seed + 20,
            reps_null=2000 // scale,
            reps_power=1000 // scale,
            n_per_group=150,
        ),
        "cox_recovery": cox_recovery(
            out, seed + 30, n_single=3000, reps=300 // scale, n_rep=600
        ),
    }
    write_json(summary, out / "summary.json")
    print(f"simulation results written to {out}")
    return summary
