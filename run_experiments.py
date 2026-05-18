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


# ---------------------------------------------------------------------------
# CLI arguments
#
# This entry point runs the fixed report suite. The CLI only controls where
# outputs go, which grids/seeds are used, whether budgets are shortened for a
# smoke test, and whether plot files are generated.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment-aligned RL experiments.")

    # Output location for the master CSV, per-group CSVs, plots, and overview
    # summaries produced by the experiment pipeline.
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/assignment_experiments"),
        help="Output directory for CSV files and plots.",
    )

    # Grid selection:
    # the first grid is the primary grid used for hyperparameter sweeps; all
    # provided grids are also included in the grid-comparison group.
    parser.add_argument(
        "--grid",
        type=Path,
        nargs="+",
        default=list(DEFAULT_GRIDS),
        help="Grid files. The first grid is used for hyperparameter setups; all grids are used for grid comparison.",
    )

    # Seed selection:
    # every experiment case is run once per seed so the overview can aggregate
    # across random initialisation, exploration, and environment stochasticity.
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds to evaluate.",
    )

    # Runtime controls:
    # ``--quick`` preserves the case structure but shrinks episode/evaluation
    # budgets, while ``--no_plots`` skips image generation after CSV writing.
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
    # ------------------------------------------------------------------
    # Resolve experiment specification
    # ------------------------------------------------------------------
    args = parse_args()
    cases = build_cases(args.grid)
    cfg = defaults(quick=args.quick)

    # ------------------------------------------------------------------
    # Run suite and write CSV outputs
    # ------------------------------------------------------------------
    # ``run_suite`` handles the nested loop over case x seed x algorithm and
    # writes both the master results.csv and each per-group results.csv.
    results = run_suite(
        cases=cases,
        base_cfg=cfg,
        out_dir=args.out_dir,
        seeds=args.seeds,
    )

    # ------------------------------------------------------------------
    # Plot and overview outputs
    # ------------------------------------------------------------------
    # Plots are optional because quick smoke tests often only need to verify
    # that training/evaluation completes and CSVs are written.
    if not args.no_plots:
        print("\nGenerating plots...")
        save_all(results, args.out_dir)

    # The overview step prints and saves aggregate Markdown/CSV summaries,
    # including seed means/stds when multiple seeds were requested.
    print_and_save_overview(results, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
