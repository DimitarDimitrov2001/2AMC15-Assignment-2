"""Assignment experiment runner.

Runs the six report-oriented experiment groups for Value Iteration,
on-policy Monte Carlo, and Q-learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overview import print_and_save_overview
from experiments.plots import save_all
from experiments.runner import run_suite
from experiments.specs import DEFAULT_GRIDS, build_cases, defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment-aligned RL experiments.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/assignment_experiments"),
        help="Output directory for CSV files and plots.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        nargs="+",
        default=list(DEFAULT_GRIDS),
        help="Grid files. The first grid is used for hyperparameter setups; all grids are used for grid comparison.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use short episode/evaluation budgets for smoke testing.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(args.grid)
    cfg = defaults(quick=args.quick)
    results = run_suite(
        cases=cases,
        base_cfg=cfg,
        out_dir=args.out_dir,
        seeds=args.seeds,
    )

    if not args.no_plots:
        print("\nGenerating plots...")
        save_all(results, args.out_dir)

    print_and_save_overview(results, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
