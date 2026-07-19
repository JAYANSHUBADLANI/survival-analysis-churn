"""Matplotlib figures shared by the experiment runners.

Uses the non-interactive Agg backend so figures render identically in
headless environments; every function draws onto a provided Axes or
returns a Figure the caller saves.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ._validation import FloatArray  # noqa: E402
from .km import KaplanMeierFit  # noqa: E402

_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")


def km_step_arrays(fit: KaplanMeierFit) -> tuple[FloatArray, FloatArray]:
    """Step-plot arrays with the (0, 1) starting point prepended."""
    t = np.concatenate(([0.0], fit.event_times))
    s = np.concatenate(([1.0], fit.survival))
    return t, s


def plot_km_fits(
    ax: plt.Axes,
    fits: dict[str, KaplanMeierFit],
    *,
    ci: bool = True,
    true_curve: tuple[FloatArray, FloatArray] | None = None,
    xlabel: str = "time",
    title: str = "",
) -> None:
    """Kaplan-Meier step curves with optional CI bands and true overlay."""
    for color, (label, fit) in zip(_COLORS, fits.items()):
        t, s = km_step_arrays(fit)
        ax.step(t, s, where="post", color=color, lw=1.8, label=label)
        if ci:
            tl = np.concatenate(([0.0], fit.event_times))
            lo = np.concatenate(([1.0], fit.ci_lower))
            hi = np.concatenate(([1.0], fit.ci_upper))
            ax.fill_between(
                tl, lo, hi, step="post", color=color, alpha=0.18, lw=0
            )
    if true_curve is not None:
        ax.plot(
            true_curve[0],
            true_curve[1],
            color="black",
            ls="--",
            lw=1.6,
            label="true S(t)",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("survival probability S(t)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


def plot_log_neg_log(
    ax: plt.Axes,
    curves: dict[object, tuple[FloatArray, FloatArray]],
    *,
    xlabel: str = "log(time)",
    title: str = "",
) -> None:
    """log(-log S) curves; parallel lines support proportional hazards."""
    for color, (label, (x, y)) in zip(_COLORS, curves.items()):
        ax.step(x, y, where="post", color=color, lw=1.6, label=str(label))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("log(-log S(t))")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)


def forest_plot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    truth: dict[str, float] | None = None,
    title: str = "",
) -> None:
    """Hazard ratios with CIs on a log axis; optional true-value markers.

    ``summary`` must have index = covariate names and columns
    ``hazard_ratio`` plus lower/upper CI columns starting with
    ``hr_lower``/``hr_upper``.
    """
    lower_col = next(c for c in summary.columns if c.startswith("hr_lower"))
    upper_col = next(c for c in summary.columns if c.startswith("hr_upper"))
    y = np.arange(len(summary))[::-1]
    hr = summary["hazard_ratio"].to_numpy()
    lo = summary[lower_col].to_numpy()
    hi = summary[upper_col].to_numpy()
    ax.errorbar(
        hr,
        y,
        xerr=np.vstack([hr - lo, hi - hr]),
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
        lw=1.6,
        label="estimate (95% CI)",
    )
    if truth is not None:
        ax.scatter(
            [truth[name] for name in summary.index],
            y,
            marker="x",
            s=70,
            color="#d62728",
            zorder=5,
            label="true HR",
        )
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xscale("log")
    ax.set_xlabel("hazard ratio (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(frameon=False)


def schoenfeld_panel(
    times: FloatArray,
    scaled_residuals: FloatArray,
    coef: FloatArray,
    names: list[str],
    p_values: FloatArray,
    *,
    xlabel: str = "time",
) -> plt.Figure:
    """Grid of scaled Schoenfeld residual scatters with linear trends.

    Each panel shows ``beta_hat_j + s*_kj`` against event time, the
    horizontal line at ``beta_hat_j`` (constant-effect reference) and a
    least-squares trend; the panel title reports the slope-test p-value.
    """
    p = len(names)
    ncols = min(3, p)
    nrows = int(np.ceil(p / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False
    )
    for j, name in enumerate(names):
        ax = axes[j // ncols][j % ncols]
        y = coef[j] + scaled_residuals[:, j]
        ax.scatter(times, y, s=8, alpha=0.35, color="#1f77b4")
        ax.axhline(coef[j], color="black", lw=1.2, ls="--")
        design = np.column_stack([np.ones_like(times), times])
        slope_fit, *_ = np.linalg.lstsq(design, y, rcond=None)
        grid = np.linspace(times.min(), times.max(), 50)
        ax.plot(
            grid,
            slope_fit[0] + slope_fit[1] * grid,
            color="#d62728",
            lw=1.6,
        )
        ax.set_title(f"{name}  (PH test p = {p_values[j]:.3g})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\hat\beta_j(t)$")
        ax.grid(alpha=0.3)
    for j in range(p, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return fig
