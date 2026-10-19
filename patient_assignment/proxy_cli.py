"""Proxy agreement and allocation analysis, isolated from inference commands."""

import argparse
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="LLM assessment against a proxy benchmark")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("proxy-analysis", help="Analyze existing categorical predictions")
    plots = commands.add_parser("proxy-plots", help="Render saved results; no solver or inference")
    for sub in (run, plots):
        sub.add_argument("--models", nargs="+", required=sub is run)
        sub.add_argument("--starting-parameters", nargs="+")
        sub.add_argument("--chunk-ids", nargs="+")
        sub.add_argument("--show_type", "--show-type", choices=("Point", "Box"), default="Point")
        sub.add_argument("--facet-cols", type=int, default=3)
        sub.add_argument("--output", type=Path, required=True)
    for name in ("config", "chunks", "profiles"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--benchmark", required=True)
    run.add_argument(
        "--objective", choices=("complexity-range", "lexicographic"), default="complexity-range"
    )
    run.add_argument(
        "--n-samples", type=int, help="Default 30; random objectives, not uniform sampling"
    )
    run.add_argument(
        "--sample-seed", type=int, help="Default 42; random tie-break coefficient seed"
    )
    run.add_argument("--selection-seeds", type=int, nargs="+")
    run.add_argument("--benchmark-seed", type=int, default=1001)
    run.add_argument(
        "--chunk-seed",
        type=int,
        default=42,
        help="One selection seed for cross-chunk performance; preserves membership",
    )
    run.add_argument(
        "--solver-seed",
        type=int,
        default=0,
        help="Gurobi algorithm seed, separate from selection coefficient seeds",
    )
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--time-limit", type=float)
    run.add_argument(
        "--optimality-tolerance",
        type=float,
        default=1e-7,
        help="Absolute certification/verification tolerance; no intended degradation",
    )
    run.add_argument("--feasibility-tolerance", type=float, default=1e-9)
    run.add_argument("--integer-tolerance", type=float, default=1e-9)
    run.add_argument("--solver-optimality-tolerance", type=float, default=1e-9)
    run.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional shared content-addressed cache; default OUTPUT/cache",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    plots.add_argument("--results", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.facet_cols < 1:
            raise ValueError("facet-cols must be >= 1")
        if args.command == "proxy-plots":
            from .proxy_plots import regenerate

            regenerate(args)
            return 0
        from .proxy_analysis import run

        return run(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
