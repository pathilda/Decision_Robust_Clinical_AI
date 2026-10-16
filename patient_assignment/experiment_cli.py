"""Opt-in experiment interface; importing this module never initializes a solver."""

from pathlib import Path

DEFAULTS = {
    "models": None,
    "chunk_ids": None,
    "profile_ids": None,
    "objectives": ["complexity-range", "lexicographic"],
    "selection": ["random"],
    "coverage": "strict",
    "samples": 30,
    "sample_seed": 42,
    "regret_bounds": "none",
    "uncertainty_source": "files",
    "uncertainty_manifest": None,
    "stress_test_manifest": None,
    "perturb_fraction": 0.10,
    "uncertainty_samples": 30,
    "uncertainty_seed": 42,
    "allow_benchmark_in_uncertainty": False,
    "nominal_slack": 0.0,
    "robust_samples": 1,
    "time_limit": None,
    "threads": 1,
    "mip_gap": 0.0,
    "solver_seed": 0,
    "optimality_tolerance": 1e-6,
    "output": None,
    "cache_dir": None,
    "resume": False,
    "dry_run": False,
}


def add_commands(commands):
    experiment = commands.add_parser(
        "assignment-experiment", help="Compare existing scores on fixed chunks and profiles"
    )
    experiment.add_argument("--config", type=Path, required=True)
    for name in (
        "chunks",
        "profiles",
        "output",
        "cache-dir",
        "uncertainty-manifest",
        "stress-test-manifest",
    ):
        experiment.add_argument("--" + name, type=str, default=None)
    for name, choices in (
        ("models", None),
        ("profile-ids", None),
        ("chunk-ids", None),
        ("objectives", ["complexity-range", "lexicographic"]),
        ("selection", ["random", "matched", "robust"]),
    ):
        experiment.add_argument("--" + name, nargs="+", action="extend", choices=choices)
    experiment.add_argument("--benchmark")
    experiment.add_argument("--benchmark-kind", choices=["proxy", "clinical"])
    for name, choices in (
        ("coverage", ["strict", "intersection"]),
        ("regret-bounds", ["none", "exact"]),
        ("uncertainty-source", ["files", "perturbations"]),
    ):
        experiment.add_argument("--" + name, choices=choices)
    for name in (
        "samples",
        "sample-seed",
        "uncertainty-samples",
        "uncertainty-seed",
        "robust-samples",
        "threads",
        "solver-seed",
    ):
        experiment.add_argument("--" + name, type=int)
    for name in (
        "perturb-fraction",
        "nominal-slack",
        "time-limit",
        "mip-gap",
        "optimality-tolerance",
    ):
        experiment.add_argument("--" + name, type=float)
    for name in ("resume", "dry-run", "allow-benchmark-in-uncertainty"):
        from argparse import BooleanOptionalAction

        experiment.add_argument("--" + name, action=BooleanOptionalAction, default=None)

    profiles = commands.add_parser(
        "make-profiles", help="Generate synthetic conserved-load profiles"
    )
    profiles.add_argument(
        "--clinician",
        nargs=3,
        action="append",
        required=True,
        metavar=("ID", "INITIAL_COUNT", "INITIAL_COMPLEXITY"),
    )
    profiles.add_argument("--clinician-capacity", type=int, required=True)
    profiles.add_argument("--imbalance-factors", nargs="+", type=float, default=[0, 0.5, 1])
    profiles.add_argument("--profile-seed", type=int, default=42)
    profiles.add_argument("--score-min", type=float, default=1)
    profiles.add_argument("--score-max", type=float, default=3)
    profiles.add_argument("--output", type=Path, required=True)

    plots = commands.add_parser("assignment-plots", help="Plot saved tables without solving")
    plots.add_argument("--results", type=Path, required=True)
    plots.add_argument("--output", type=Path, required=True)
    plots.add_argument(
        "--plots",
        nargs="+",
        choices=["regret-envelope", "profile-effect", "assignment-regret", "robustness"],
        default=["regret-envelope", "profile-effect", "assignment-regret", "robustness"],
    )
    plots.add_argument("--format", nargs="+", choices=["png", "pdf"], default=["png"])
    for name in ("models", "profile-ids", "objectives", "selection"):
        plots.add_argument("--" + name, nargs="+")
    plots.add_argument("--max-instances-per-figure", type=int, default=12)
    plots.add_argument("--plot-seed", type=int, default=42)
    plots.add_argument("--regret-threshold", type=float)
    plots.add_argument("--disagreement-threshold", type=float)


def execute(args):
    if args.command == "make-profiles":
        from .experiment_inputs import generate_profiles, save_yaml

        save_yaml(args.output, generate_profiles(args))
        return 0
    if args.command == "assignment-plots":
        from .experiment_plots import plot_results

        return plot_results(args)
    from .experiment import run_experiment
    from .experiment_inputs import prepare

    return run_experiment(prepare(args))
