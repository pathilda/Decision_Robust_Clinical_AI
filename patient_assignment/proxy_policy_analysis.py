"""Uncertainty-policy orchestration; the legacy deterministic runner stays separate."""

# Cache callbacks execute synchronously before the enclosing loop advances.
# ruff: noqa: B023
import json
import os
import tempfile
from contextlib import ExitStack
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import pandas as pd

from . import experiment_solver as solver
from . import proxy_policy_solver as policy_solver
from .experiment_inputs import digest, number
from .inputs import file_hash
from .optimization import SolverSession
from .proxy_analysis import (
    ALLOCATION_COLUMNS,
    RESULT_COLUMNS,
    allocation_rows,
    run_identity,
    versions,
)
from .proxy_inputs import prepare
from .proxy_solver import ProxySettings, canonical, evaluate, select, stream_seed
from .proxy_uncertainty import configuration, load_probabilities
from .reporting import fresh_directory, write_csv


def write_json(path, payload):
    """Publish configuration/manifests atomically so interrupted exports remain resumable."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)
        temporary = handle.name
    os.replace(temporary, path)


class PolicyCache:
    """Only complete certified objects become visible; interrupted temporary files are ignored."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def get(self, identity, compute):
        path = self.path / (digest(identity) + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["identity"] != identity or saved["payload_hash"] != digest(saved["payload"]):
                raise ValueError("Corrupt policy cache record")
            self.hits += 1
            return saved["payload"]
        self.misses += 1
        try:
            payload = compute()
        except Exception as exc:
            # Solver API errors (including restricted-license size limits) are failed cells,
            # not a reason to lose the remainder of an experiment. Do not hide code errors.
            if type(exc).__module__.split(".")[0] == "gurobipy":
                raise RuntimeError(
                    f"Gurobi solve failed (code {getattr(exc, 'errno', 'unknown')}); check solver/license limits"
                ) from exc
            raise
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.path, suffix=".tmp", encoding="utf-8", delete=False
        ) as handle:
            json.dump(
                {"identity": identity, "payload": payload, "payload_hash": digest(payload)},
                handle,
                allow_nan=False,
            )
            temporary = handle.name
        os.replace(temporary, path)
        return payload


def problem_for(cell, instance, mid, nominal, probabilities, class_weights):
    metadata = dict(cell)
    problem = {"policy": cell["policy"]}
    if cell["policy"] == "stochastic":
        data = policy_solver.scenarios(
            probabilities,
            class_weights,
            mid,
            instance["chunk_id"],
            cell["scenario_seed"],
            cell["scenario_count"],
            cell["temperature"],
        )
        problem["scenarios"] = data
        metadata.update(
            scenario_fingerprint=data["fingerprint"],
            scenario_generator=data["generator"],
            scenario_weighting="uniform_empirical",
        )
    elif cell["policy"] == "robust":
        n = len(instance["patients"])
        budget = cell.get(
            "gamma_count",
            int(
                (Decimal(str(cell.get("gamma_fraction", 0))) * n).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )
        if budget > n:
            raise ValueError("gamma_count exceeds chunk patient count")
        supports = policy_solver.support(
            nominal, probabilities, class_weights, cell["probability_threshold"]
        )
        problem.update(supports=supports, budget=budget)
        metadata.update(gamma_effective=budget, uncertainty_support_fingerprint=digest(supports))
    return problem, metadata


def summaries(rows, output):
    frame = pd.DataFrame(rows)
    group = ["model_id", "policy", "policy_configuration_id", "chunk_id", "profile_id"]
    result = []
    for identity, all_rows in frame[frame.is_seed_sample].groupby(group, sort=True):
        good = all_rows[all_rows.certified]
        row = dict(
            zip(group, identity),
            n_total=len(all_rows),
            n_success=len(good),
            n_unique=good.solution_hash.nunique() if len(good) else 0,
        )
        row["n_unique_assignments"] = row["n_unique"]
        row["duplicate_rate"] = 1 - row["n_unique"] / len(good) if len(good) else None
        for metric in ("proxy_regret", "exact_assignment_pct", "holdout_mean", "holdout_cvar"):
            if metric not in good or not len(good):
                continue
            values = good[metric].dropna()
            if values.empty:
                continue
            row.update(
                {
                    metric + "_" + name: value
                    for name, value in {
                        "mean": values.mean(),
                        "median": values.median(),
                        "sd": values.std(ddof=0),
                        "min": values.min(),
                        "max": values.max(),
                        **{
                            "q" + str(int(q * 100)).zfill(2): values.quantile(q)
                            for q in (0.05, 0.25, 0.75, 0.95)
                        },
                    }.items()
                }
            )
        row["regret_nonzero_fraction"] = (
            float((good.proxy_regret > 1e-7).mean()) if len(good) else None
        )
        result.append(row)
    write_csv(output / "policy_summary.csv", result, group + ["n_total", "n_success", "n_unique"])
    good = frame[frame.certified & frame.is_seed_sample]
    paired = []
    baseline = good[good.policy == "deterministic"]
    for row in good[good.policy != "deterministic"].to_dict("records"):
        match = baseline
        for k in ("model_id", "chunk_id", "profile_id", "selection_seed"):
            match = match[match[k] == row[k]]
        if len(match) == 1:
            base = match.iloc[0]
            paired.append(
                {k: row[k] for k in group + ["selection_seed"]}
                | {
                    "proxy_regret_difference": row["proxy_regret"] - base.proxy_regret,
                    "assignment_agreement_difference": row["exact_assignment_pct"]
                    - base.exact_assignment_pct,
                    "interpretation": "descriptive paired solver outputs, not independent clinical observations",
                }
            )
    write_csv(output / "paired_policy_differences.csv", paired, group)
    replicates = []
    for row in result:
        config = next(
            r for r in rows if r["policy_configuration_id"] == row["policy_configuration_id"]
        )
        if row["policy"] == "stochastic":
            replicates.append(
                row | {k: config.get(k) for k in ("scenario_count", "scenario_seed", "temperature")}
            )
    write_csv(output / "scenario_replicate_summary.csv", replicates, group + ["scenario_seed"])


def run(args):
    prepared = prepare(args)
    resolved = prepared["resolved"]
    config, grid = configuration(
        args.uncertainty_config, args.policies or ["deterministic"], args.policy_configuration_ids
    )
    needs_probabilities = (
        any(c["policy"] != "deterministic" for c in grid)
        or config["evaluation"]["holdout_scenarios"] > 0
    )
    probabilities, diagnostics, sources = (
        load_probabilities(args, prepared, config) if needs_probabilities else ({}, [], {})
    )
    tolerance = number(args.optimality_tolerance, "optimality-tolerance", 1e-9)
    if tolerance > 1e-3:
        raise ValueError("optimality-tolerance must be <= .001")
    settings = ProxySettings(
        time_limit=args.time_limit,
        threads=args.threads,
        seed=args.solver_seed,
        feasibility_tolerance=args.feasibility_tolerance,
        integer_tolerance=args.integer_tolerance,
        solver_optimality_tolerance=args.solver_optimality_tolerance,
    )
    settings.validate()
    resolved.update(
        uncertainty=config,
        policy_grid=grid,
        probability_sources=sources,
        versions=versions(),
        solver_parameters=settings.parameters(),
        implementation_hash=digest(
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        seed_roles={
            "selection": "paired random coefficient family independent of policy",
            "scenario": "training only; independent of profile and selection",
            "solver": args.solver_seed,
            "benchmark": args.benchmark_seed,
            "holdout": "separate role-derived RNG, shared across all policy parameters",
        },
        assumptions=[
            "Independent patient categorical marginals; not calibrated clinical risk",
            "Fixed point proxy is not ground truth",
            "Random linear objectives are not uniform optimal-assignment sampling",
            "Discrete cardinality-budget robust set, not a probability guarantee",
        ],
        intended_objective_degradation=0,
    )
    seeds = resolved["selection_seeds"]
    all_seeds = list(dict.fromkeys([*seeds, args.chunk_seed]))
    feasible = sum(i["feasible"] for i in prepared["instances"])
    cells = len(prepared["instances"]) * len(args.models) * len(grid)
    if any(
        c.get("gamma_count", 0) > len(i["patients"]) for c in grid for i in prepared["instances"]
    ):
        raise ValueError("gamma_count exceeds chunk patient count")
    stages = 2 if args.objective == "lexicographic" else 1
    planned = {
        "policy_cells": cells,
        "selection_attempts": cells * len(all_seeds),
        "benchmark_optima": feasible,
        "benchmark_selections": feasible,
        "policy_optima": feasible * len(args.models) * len(grid),
        "policy_selections": feasible * len(args.models) * len(grid) * len(all_seeds),
        "minimum_solves": feasible
        * (stages + 1 + len(args.models) * len(grid) * (stages + len(all_seeds))),
        "additional_solves": "Robust separation iterations in optimum AND random selection",
        "large_grid_warning": cells * len(all_seeds) > 10000,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned": planned,
                    "policy_grid": grid,
                    "probability_validation": diagnostics,
                    "feasible_instances": feasible,
                    "total_instances": len(prepared["instances"]),
                },
                indent=2,
            )
        )
        return 0
    identity = run_identity(resolved)
    output = args.output.resolve()
    if args.resume:
        previous = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        fingerprint = previous.pop("run_fingerprint", None)
        if fingerprint != identity or run_identity(previous) != identity:
            raise ValueError(
                "Incompatible resume: probability, policy, input, implementation or solver changed"
            )
    else:
        fresh_directory(output)
    resolved["run_fingerprint"] = identity
    write_json(output / "resolved_config.json", resolved)
    cache = PolicyCache(args.cache_dir or output / "cache")
    rows, allocations, references, benchmarks, optima = [], [], [], [], []
    class_weights = [resolved["category_weights"][c] for c in resolved["category_order"]]
    ev = config["evaluation"]
    with ExitStack() as stack:
        active = None

        def session():
            nonlocal active
            if active is None:
                active = stack.enter_context(SolverSession(settings))
            return active

        def base_key(instance, mid, weights):
            return {
                "algorithm": policy_solver.ALGORITHM,
                "implementation": resolved["implementation_hash"],
                "versions": resolved["versions"],
                "instance": canonical(instance),
                "model": mid,
                "weights": weights,
                "objective": args.objective,
                "solver": settings.parameters(),
                "tolerance": tolerance,
            }

        for instance in prepared["instances"]:
            context = {k: instance[k] for k in ("chunk_id", "profile_id")}
            bw = {p: prepared["loaded"][args.benchmark]["weights"][p] for p in instance["patients"]}
            bk = base_key(instance, args.benchmark, bw)
            bo = reference = None
            reference_id = digest({"kind": "reference", "base": bk, "seed": args.benchmark_seed})
            error = None
            try:
                if not instance["feasible"]:
                    raise ValueError(
                        "INFEASIBLE: chunk exceeds remaining clinician count capacities"
                    )
                bo = cache.get(
                    {"kind": "nominal", "base": bk},
                    lambda: solver.nominal(
                        session(), canonical(instance), bw, args.objective, tolerance
                    ),
                )
                reference = cache.get(
                    {"kind": "reference", "base": bk, "seed": args.benchmark_seed},
                    lambda: select(
                        session(), instance, bw, bo, tolerance, args.benchmark_seed, "benchmark"
                    ),
                )
                solver.verify(instance, bw, reference["assignment"], bo, tolerance)
                references.extend(
                    allocation_rows(
                        reference["assignment"],
                        reference_id,
                        args.benchmark,
                        instance,
                        args.benchmark_seed,
                        "reference",
                        args.benchmark,
                    )
                )
                benchmarks.append(
                    dict(
                        **context,
                        benchmark_id=args.benchmark,
                        proxy_optimum=bo["g"],
                        count_optimum=bo["d"],
                        reference_solution_id=reference_id,
                        reference_seed=args.benchmark_seed,
                        nominal_stages=bo["stages"],
                        certified=True,
                    )
                )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
                benchmarks.append(dict(**context, certified=False, failure_reason=error))
            for mid in args.models:
                nominal = {p: prepared["loaded"][mid]["weights"][p] for p in instance["patients"]}
                probs = (
                    {p: probabilities[mid][p] for p in instance["patients"]}
                    if mid in probabilities
                    else {}
                )
                holdout = (
                    policy_solver.scenarios(
                        probs,
                        class_weights,
                        mid,
                        instance["chunk_id"],
                        ev["holdout_seed"],
                        ev["holdout_scenarios"],
                        role="holdout",
                    )
                    if ev["holdout_scenarios"]
                    else None
                )
                for cell in grid:
                    problem, metadata = problem_for(
                        cell, instance, mid, nominal, probs, class_weights
                    )
                    key = {
                        "kind": "policy",
                        "base": base_key(instance, mid, nominal),
                        "problem": problem,
                        "configuration": cell,
                        "probability_source": sources.get(mid),
                    }
                    optimum, cell_error = None, error
                    optrow = dict(model_id=mid, **context, **metadata)
                    try:
                        if error:
                            raise ValueError(error)
                        optimum = cache.get(
                            key,
                            lambda: (
                                solver.nominal(
                                    session(),
                                    canonical(instance),
                                    nominal,
                                    args.objective,
                                    tolerance,
                                )
                                if cell["policy"] == "deterministic"
                                else policy_solver.solve(
                                    session(), instance, nominal, problem, args.objective, tolerance
                                )
                            ),
                        )
                        if not optimum["certified"] or not all(
                            s["certified"] for s in optimum["stages"]
                        ):
                            raise ValueError("Uncertified cached policy optimum")
                        verified_value = (
                            solver.actual(instance, nominal, optimum["assignment"])["range"]
                            if cell["policy"] == "deterministic"
                            else policy_solver.objective(
                                instance, nominal, problem, optimum["assignment"]
                            )
                        )
                        if abs(verified_value - optimum["g"]) > tolerance:
                            raise ValueError(
                                "Cached policy optimum fails independent objective verification"
                            )
                        optrow.update(
                            policy_optimum=optimum["g"],
                            count_optimum=optimum["d"],
                            certified=True,
                            optimization_status="OPTIMAL",
                            bound=optimum["stages"][-1]["bound"],
                            incumbent=optimum["g"],
                            gap=optimum["stages"][-1]["gap"],
                            solver_walltime=sum(s["runtime"] for s in optimum["stages"]),
                            separation_iterations=optimum.get("separation_iterations", 0),
                            termination_reason=optimum.get(
                                "termination_reason", "certified_nominal_optimum"
                            ),
                            diagnostics=json.dumps(optimum["stages"]),
                        )
                    except (ValueError, RuntimeError) as exc:
                        cell_error = str(exc)
                        optrow.update(
                            certified=False,
                            optimization_status="UNCERTIFIED",
                            failure_reason=cell_error,
                            diagnostics=json.dumps(getattr(exc, "diagnostics", None)),
                        )
                    optima.append(optrow)
                    for seed in all_seeds:
                        skey = {"kind": "policy_selection", "optimum": key, "seed": seed}
                        row = dict.fromkeys(
                            [
                                "scenario_count",
                                "scenario_seed",
                                "scenario_fingerprint",
                                "temperature",
                                "gamma_fraction",
                                "gamma_effective",
                                "probability_threshold",
                                "uncertainty_support_fingerprint",
                                "selected_policy_objective",
                                "expected_scenario_objective",
                                "robust_worst_case_objective",
                                "model_point_objective",
                                "count_range",
                                "holdout_mean",
                                "holdout_cvar",
                            ]
                        )
                        row.update(
                            model_id=mid,
                            **context,
                            **metadata,
                            benchmark_id=args.benchmark,
                            benchmark_kind="proxy",
                            objective=args.objective,
                            n_patients=len(nominal),
                            selection_seed=seed,
                            solver_seed=args.solver_seed,
                            selection_stream_seed=str(stream_seed(instance, seed)),
                            reference_seed=args.benchmark_seed,
                            sample_index=seeds.index(seed) if seed in seeds else None,
                            is_seed_sample=seed in seeds,
                            is_chunk_evaluation=seed == args.chunk_seed,
                            solution_id=digest(skey),
                            reference_solution_id=reference_id,
                            proxy_optimum=bo["g"] if bo else None,
                            policy_optimum=optimum["g"] if optimum else None,
                            nominal_optimum=optimum["g"]
                            if optimum and cell["policy"] == "deterministic"
                            else None,
                            optimization_status=optrow["optimization_status"],
                            certified=False,
                            n_holdout_scenarios=ev["holdout_scenarios"],
                            holdout_seed=ev["holdout_seed"] if holdout else None,
                            holdout_fingerprint=holdout["fingerprint"] if holdout else None,
                            run_fingerprint=identity,
                        )
                        try:
                            if cell_error:
                                raise ValueError(cell_error)
                            selected = cache.get(
                                skey,
                                lambda: (
                                    select(session(), instance, nominal, optimum, tolerance, seed)
                                    if cell["policy"] == "deterministic"
                                    else policy_solver.solve(
                                        session(),
                                        instance,
                                        nominal,
                                        problem,
                                        args.objective,
                                        tolerance,
                                        optimum,
                                        seed,
                                    )
                                ),
                            )
                            chosen = selected["assignment"]
                            actual = solver.actual(instance, nominal, chosen)
                            achieved = (
                                actual["range"]
                                if cell["policy"] == "deterministic"
                                else policy_solver.objective(instance, nominal, problem, chosen)
                            )
                            if (
                                not selected["certified"]
                                or abs(achieved - optimum["g"]) > tolerance
                            ):
                                raise ValueError("Selected assignment not certified policy-optimal")
                            if bo["d"] is not None and actual["count_range"] != bo["d"]:
                                raise ValueError(
                                    "Candidate and common benchmark count optima differ"
                                )
                            metrics = evaluate(instance, bw, bo, chosen, reference["assignment"])
                            if metrics["proxy_regret"] < -tolerance:
                                raise ValueError(
                                    "Negative proxy regret below certified benchmark optimum"
                                )
                            row.update(
                                **metrics,
                                solution_hash=digest(chosen),
                                certified=True,
                                solver_status="OPTIMAL",
                                selection_status="OPTIMAL",
                                selected_policy_objective=achieved,
                                expected_scenario_objective=achieved
                                if cell["policy"] == "stochastic"
                                else None,
                                robust_worst_case_objective=achieved
                                if cell["policy"] == "robust"
                                else None,
                                model_point_objective=actual["range"],
                                count_range=actual["count_range"],
                                initial_workloads=json.dumps(instance["clinicians"]),
                                final_workloads=json.dumps(actual["loads"], default=str),
                                assignment_disagreement_to_reference_pct=100
                                - metrics["exact_assignment_pct"],
                                optimality_gap=selected["stages"][-1]["gap"],
                                selection_diagnostics=json.dumps(selected["stages"]),
                            )
                            if holdout:
                                row.update(
                                    policy_solver.holdout_metrics(
                                        instance, holdout, chosen, ev["cvar_alpha"]
                                    )
                                )
                            role = (
                                "seed_sample+chunk_evaluation"
                                if seed in seeds and seed == args.chunk_seed
                                else ("seed_sample" if seed in seeds else "chunk_evaluation")
                            )
                            allocations.extend(
                                a | metadata
                                for a in allocation_rows(
                                    chosen,
                                    row["solution_id"],
                                    mid,
                                    instance,
                                    seed,
                                    role,
                                    args.benchmark,
                                )
                            )
                        except (ValueError, RuntimeError) as exc:
                            row.update(
                                certified=False,
                                solver_status="FAILED",
                                selection_status="UNCERTIFIED",
                                failure_reason=str(exc),
                                selection_diagnostics=json.dumps(getattr(exc, "diagnostics", None)),
                            )
                        rows.append(row)
    failures = [r for r in rows if not r["certified"]]
    report = {
        "planned_solve_counts": planned,
        "expected_allocations": len(rows),
        "successful_allocations": len(rows) - len(failures),
        "failed_allocations": len(failures),
        "total_instances": len(prepared["instances"]),
        "feasible_instances": feasible,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "benchmark_id": args.benchmark,
    }
    tables = {
        "accuracy_vs_proxy.csv": prepared["accuracy"],
        "results.csv": rows,
        "allocations.csv": allocations,
        "reference_allocations.csv": references,
        "policy_grid.csv": grid,
        "policy_optima.csv": optima,
        "probability_validation.csv": diagnostics,
        "run_failures.csv": failures,
        "holdout_metrics.csv": [
            {
                k: v
                for k, v in r.items()
                if k.startswith("holdout")
                or k
                in (
                    "solution_id",
                    "model_id",
                    "policy",
                    "policy_configuration_id",
                    "chunk_id",
                    "profile_id",
                    "selection_seed",
                    "scenario_seed",
                    "n_holdout_scenarios",
                    "cvar_alpha",
                )
            }
            for r in rows
            if r["certified"] and ev["holdout_scenarios"]
        ],
    }
    for name, data in tables.items():
        columns = (
            RESULT_COLUMNS
            if name in ("results.csv", "run_failures.csv")
            else ALLOCATION_COLUMNS
            if "allocations" in name
            else ["model_id"]
            if not data
            else []
        )
        if name == "allocations.csv":
            columns = [*columns, "policy", "policy_configuration_id", "scenario_seed"]
        if name == "holdout_metrics.csv":
            columns = [
                "solution_id",
                "model_id",
                "policy",
                "policy_configuration_id",
                "chunk_id",
                "profile_id",
                "selection_seed",
                "scenario_seed",
                "holdout_seed",
                "holdout_fingerprint",
                "n_holdout_scenarios",
                "holdout_mean",
                "holdout_std",
                "holdout_max",
                "holdout_median",
                "holdout_q90",
                "holdout_q95",
                "holdout_cvar",
                "cvar_alpha",
            ]
        write_csv(output / name, data, columns)
    summaries(rows, output)
    write_json(output / "benchmark_optima.json", benchmarks)
    write_json(output / "validation_report.json", report)
    names = [
        *tables,
        "policy_summary.csv",
        "paired_policy_differences.csv",
        "scenario_replicate_summary.csv",
        "benchmark_optima.json",
        "validation_report.json",
        "resolved_config.json",
    ]
    write_json(output / "artifacts.json", {name: file_hash(output / name) for name in names})
    from .proxy_policy_plots import render

    render(output, output, args.show_type, args.facet_cols, allow_existing=True)
    print(
        f"{report['successful_allocations']}/{len(rows)} certified policy allocations: {output / 'index.html'}"
    )
    return 2 if failures else 0
