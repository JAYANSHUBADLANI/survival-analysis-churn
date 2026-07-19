"""Command-line entry point.

Usage::

    python -m survival run-simulation [--outdir results] [--seed 2026]
                                      [--fast]
    python -m survival run-telco      [--outdir results]
                                      [--data data/telco_churn.csv]
"""
from __future__ import annotations

import argparse
import sys

from .experiments import run_simulation, run_telco


def build_parser() -> argparse.ArgumentParser:
    """CLI parser with the two experiment subcommands."""
    parser = argparse.ArgumentParser(
        prog="survival",
        description=(
            "Survival analysis for customer churn, from scratch: "
            "validation on simulated data with known truth, and analysis "
            "of the IBM Telco churn cohort."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser(
        "run-simulation",
        help="Layer-1 validation against simulated data with known truth",
    )
    p_sim.add_argument(
        "--outdir", default="results", help="output root (default: results)"
    )
    p_sim.add_argument(
        "--seed", type=int, default=2026, help="base RNG seed (default: 2026)"
    )
    p_sim.add_argument(
        "--fast",
        action="store_true",
        help="reduce replication counts ~10x (smoke runs)",
    )

    p_telco = sub.add_parser(
        "run-telco", help="Layer-2 analysis of the IBM Telco churn data"
    )
    p_telco.add_argument(
        "--outdir", default="results", help="output root (default: results)"
    )
    p_telco.add_argument(
        "--data",
        default="data/telco_churn.csv",
        help="path to the Telco CSV (default: data/telco_churn.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the requested experiment runner."""
    args = build_parser().parse_args(argv)
    if args.command == "run-simulation":
        run_simulation.main(
            outdir=args.outdir, seed=args.seed, fast=args.fast
        )
    else:
        run_telco.main(outdir=args.outdir, data_path=args.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
