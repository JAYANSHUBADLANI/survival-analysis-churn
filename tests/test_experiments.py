"""End-to-end runners: produce every artifact and sane headline numbers."""
import json
from pathlib import Path

import pytest

from survival.__main__ import build_parser
from survival.experiments import run_simulation, run_telco

DATA = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"


def test_cli_parser_wires_both_subcommands():
    parser = build_parser()
    args = parser.parse_args(["run-simulation", "--fast", "--seed", "7"])
    assert args.command == "run-simulation"
    assert args.fast and args.seed == 7
    args = parser.parse_args(["run-telco", "--data", "x.csv"])
    assert args.command == "run-telco"
    assert args.data == "x.csv"


def test_simulation_runner_fast_mode(tmp_path):
    summary = run_simulation.main(
        outdir=str(tmp_path), seed=2026, fast=True
    )
    out = tmp_path / "simulation"
    for name in (
        "km_recovery.png",
        "greenwood_coverage.png",
        "logrank_calibration.png",
        "cox_recovery.png",
        "km_recovery.csv",
        "cox_recovery_single.md",
        "cox_recovery_repeated.csv",
        "summary.json",
    ):
        assert (out / name).exists(), name
    assert summary["km_recovery"]["max_abs_deviation"] < 0.05
    assert 0.25 <= summary["km_recovery"]["censored_fraction"] <= 0.45
    # Loose bands: fast mode uses ~10x fewer replicates.
    for cov in summary["greenwood_coverage"]["coverage_by_time"].values():
        assert 0.85 <= cov <= 1.0
    assert summary["logrank_calibration"]["type1_error"] <= 0.12
    assert summary["cox_recovery"]["repeated"]["max_abs_bias"] < 0.1
    reloaded = json.loads((out / "summary.json").read_text())
    assert reloaded["seed"] == 2026


@pytest.mark.skipif(not DATA.exists(), reason="Telco CSV not present")
def test_telco_runner_end_to_end(tmp_path):
    summary = run_telco.main(outdir=str(tmp_path), data_path=str(DATA))
    out = tmp_path / "telco"
    for name in (
        "km_by_contract.png",
        "loglog_by_contract.png",
        "schoenfeld_full.png",
        "cox_forest.png",
        "km_by_contract.csv",
        "logrank_contract.md",
        "cox_full.csv",
        "cox_stratified.csv",
        "ph_test_full.md",
        "summary.json",
    ):
        assert (out / name).exists(), name
    cohort = summary["cohort"]
    assert cohort["n"] == 7032  # 7043 minus the 11 zero-tenure rows
    assert cohort["n_events"] == 1869
    # Known structure of this dataset: contract type separates retention
    # overwhelmingly, so the log-rank statistic is enormous.
    assert summary["logrank"]["overall_chi2"] > 100
    assert summary["logrank"]["overall_p"] < 1e-10
    hrs = summary["cox"]["full_model"]["hazard_ratios"]
    assert hrs["two_year_contract"]["hr"] < hrs["one_year_contract"]["hr"] < 1.0
    assert hrs["fiber_optic"]["hr"] > 1.0
