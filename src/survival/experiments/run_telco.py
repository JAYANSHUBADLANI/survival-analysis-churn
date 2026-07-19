"""Layer-2 analysis: IBM Telco Customer Churn cohort.

Time-to-event setup: ``tenure`` (months with the company) is the
follow-up time, ``Churn == "Yes"`` is the event, and customers still
subscribed at the data snapshot are right-censored - exactly the
administrative censoring survival methods are built for.

Pipeline: cohort summary (with reverse-KM median follow-up) ->
Kaplan-Meier by contract type with 95% CIs -> log-rank tests (overall
and pairwise) -> Cox model on six covariates -> proportional-hazards
diagnostics (scaled Schoenfeld slope tests and log(-log S) curves) ->
contract-stratified Cox refit as the remedy for the PH violation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..cox import fit_cox_ph
from ..diagnostics import (
    log_neg_log_curves,
    proportional_hazards_test,
    schoenfeld_residuals,
)
from ..km import fit_kaplan_meier
from ..logrank import logrank_test
from ..plotting import (
    forest_plot,
    plot_km_fits,
    plot_log_neg_log,
    schoenfeld_panel,
)
from ._output import ensure_dir, save_figure, write_json, write_table

COVARIATES = [
    "one_year_contract",
    "two_year_contract",
    "fiber_optic",
    "electronic_check",
    "senior_citizen",
    "paperless_billing",
]
KM_MILESTONES = (12.0, 24.0, 48.0)


def load_telco(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and clean the Telco CSV.

    Drops the 11 customers with ``tenure == 0`` (joined within the
    snapshot month, so they have no follow-up exposure; none of them
    churned). Returns the cleaned frame plus a cleaning report.
    """
    raw = pd.read_csv(path)
    n_raw = len(raw)
    df = raw[raw["tenure"] > 0].copy()
    df["time"] = df["tenure"].astype(float)
    df["event"] = (df["Churn"] == "Yes").astype(int)
    df["one_year_contract"] = (df["Contract"] == "One year").astype(float)
    df["two_year_contract"] = (df["Contract"] == "Two year").astype(float)
    df["fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(float)
    df["electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(float)
    df["senior_citizen"] = df["SeniorCitizen"].astype(float)
    df["paperless_billing"] = (df["PaperlessBilling"] == "Yes").astype(float)

    followup = fit_kaplan_meier(df["time"], 1 - df["event"])
    report = {
        "n_raw": n_raw,
        "n_dropped_zero_tenure": n_raw - len(df),
        "n": len(df),
        "n_events": int(df["event"].sum()),
        "censored_fraction": float(1 - df["event"].mean()),
        "median_followup_months_reverse_km": followup.median_survival_time(),
        "max_followup_months": float(df["time"].max()),
    }
    return df, report


def km_by_contract(df: pd.DataFrame, outdir: Path) -> dict:
    """KM curves by contract type with CIs, milestones and medians."""
    fits = {
        label: fit_kaplan_meier(sub["time"], sub["event"])
        for label, sub in df.groupby("Contract")
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_km_fits(
        ax,
        fits,
        xlabel="tenure (months)",
        title="Customer retention by contract type (Kaplan-Meier, 95% CI)",
    )
    save_figure(fig, outdir / "km_by_contract.png")

    rows = []
    summary: dict[str, dict] = {}
    for label, fit in fits.items():
        surv = fit.survival_at(np.array(KM_MILESTONES))
        lo, hi = fit.ci_at(np.array(KM_MILESTONES))
        median = fit.median_survival_time()
        rows.append(
            {
                "contract": label,
                "n": fit.n_subjects,
                "events": fit.n_events,
                **{
                    f"S({int(m)}mo)": s
                    for m, s in zip(KM_MILESTONES, surv)
                },
                "median_months": median if np.isfinite(median) else np.nan,
            }
        )
        summary[label] = {
            "n": fit.n_subjects,
            "events": fit.n_events,
            "survival_at_12mo": [
                float(surv[0]),
                [float(lo[0]), float(hi[0])],
            ],
            "median_months": None
            if not np.isfinite(median)
            else float(median),
        }
    table = pd.DataFrame(rows)
    write_table(table, outdir, "km_by_contract", index=False)
    return summary


def logrank_tables(df: pd.DataFrame, outdir: Path) -> dict:
    """Overall 3-group log-rank plus pairwise contrasts."""
    overall = logrank_test(df["time"], df["event"], df["Contract"])
    rows = [
        {
            "comparison": "all contract types",
            "chi2": overall.statistic,
            "df": overall.df,
            "p": overall.p_value,
        }
    ]
    labels = sorted(df["Contract"].unique())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sub = df[df["Contract"].isin([labels[i], labels[j]])]
            res = logrank_test(sub["time"], sub["event"], sub["Contract"])
            rows.append(
                {
                    "comparison": f"{labels[i]} vs {labels[j]}",
                    "chi2": res.statistic,
                    "df": res.df,
                    "p": res.p_value,
                }
            )
    table = pd.DataFrame(rows)
    write_table(table, outdir, "logrank_contract", index=False)
    write_table(overall.to_frame(), outdir, "logrank_contract_oe",
                index=False)
    return {
        "overall_chi2": float(overall.statistic),
        "overall_df": int(overall.df),
        "overall_p": float(overall.p_value),
    }


def cox_models(df: pd.DataFrame, outdir: Path) -> dict:
    """Full Cox fit, PH diagnostics, and the contract-stratified remedy."""
    X = df[COVARIATES]
    fit = fit_cox_ph(X, df["time"], df["event"])
    write_table(fit.summary_frame(), outdir, "cox_full",
                float_format="{:.4g}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    forest_plot(
        ax,
        fit.summary_frame(),
        title="Telco churn hazard ratios (Cox PH, Efron ties)",
    )
    save_figure(fig, outdir / "cox_forest.png")

    ph = proportional_hazards_test(fit, X, df["time"], df["event"])
    write_table(ph.table, outdir, "ph_test_full", float_format="{:.4g}")
    violations = ph.violations()

    times, resid = schoenfeld_residuals(fit, X, df["time"], df["event"])
    scaled = resid.shape[0] * (resid @ fit.covariance)
    fig = schoenfeld_panel(
        times,
        scaled,
        fit.coef,
        fit.names,
        ph.table.loc[fit.names, "p"].to_numpy(),
        xlabel="tenure (months)",
    )
    save_figure(fig, outdir / "schoenfeld_full.png")

    curves = log_neg_log_curves(df["time"], df["event"], df["Contract"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_log_neg_log(
        ax,
        curves,
        xlabel="log(tenure in months)",
        title=(
            "log(-log S(t)) by contract type - parallel lines would "
            "support PH"
        ),
    )
    save_figure(fig, outdir / "loglog_by_contract.png")

    strat_covs = [
        c for c in COVARIATES
        if c not in ("one_year_contract", "two_year_contract")
    ]
    strat = fit_cox_ph(
        df[strat_covs], df["time"], df["event"], strata=df["Contract"]
    )
    write_table(strat.summary_frame(), outdir, "cox_stratified",
                float_format="{:.4g}")
    ph_strat = proportional_hazards_test(
        strat, df[strat_covs], df["time"], df["event"],
        strata=df["Contract"],
    )
    write_table(ph_strat.table, outdir, "ph_test_stratified",
                float_format="{:.4g}")

    def hr_dict(f) -> dict:
        return {
            name: {
                "hr": float(h),
                "ci": [float(lo), float(hi)],
                "p": float(p),
            }
            for name, h, lo, hi, p in zip(
                f.names, f.hazard_ratio, f.hr_ci_lower,
                f.hr_ci_upper, f.p_values,
            )
        }

    return {
        "full_model": {
            "n": fit.n,
            "n_events": fit.n_events,
            "lr_statistic": fit.lr_statistic,
            "lr_p": fit.lr_p_value,
            "hazard_ratios": hr_dict(fit),
        },
        "ph_violations_full": violations,
        "ph_global_p_full": float(ph.table.loc["GLOBAL", "p"]),
        "stratified_model": {
            "strata": "Contract",
            "hazard_ratios": hr_dict(strat),
        },
        "ph_violations_stratified": ph_strat.violations(),
        "ph_global_p_stratified": float(
            ph_strat.table.loc["GLOBAL", "p"]
        ),
    }


def main(
    *, outdir: str = "results", data_path: str = "data/telco_churn.csv"
) -> dict:
    """Run the full Telco pipeline and write all figures/tables."""
    out = ensure_dir(Path(outdir) / "telco")
    df, report = load_telco(data_path)
    write_table(
        pd.DataFrame([report]), out, "cohort_summary", index=False
    )
    summary = {
        "cohort": report,
        "km_by_contract": km_by_contract(df, out),
        "logrank": logrank_tables(df, out),
        "cox": cox_models(df, out),
    }
    write_json(summary, out / "summary.json")
    print(f"telco results written to {out}")
    return summary
