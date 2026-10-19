"""Checkpointed proxy workflow; successful nominal and selection solves are immutable."""

import json
import platform
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import experiment_solver as solver
from .experiment import ArtifactCache
from .experiment_inputs import digest, number
from .inputs import file_hash
from .optimization import SolverSession
from .proxy_inputs import prepare
from .proxy_solver import ProxySettings, canonical, evaluate, select
from .reporting import fresh_directory, write_csv, write_json

RESULT_COLUMNS = [
    "solution_id",
    "model_id",
    "benchmark_id",
    "benchmark_kind",
    "objective",
    "chunk_id",
    "n_patients",
    "profile_id",
    "selection_seed",
    "sample_index",
    "reference_solution_id",
    "reference_seed",
    "solution_hash",
    "nominal_optimum",
    "proxy_optimum",
    "proxy_evaluated_range",
    "proxy_regret",
    "exact_assignment_pct",
    "solver_status",
    "optimality_gap",
    "is_seed_sample",
    "is_chunk_evaluation",
    "failure_reason",
]
ALLOCATION_COLUMNS = [
    "solution_id",
    "patient_id",
    "clinician_id",
    "model_id",
    "benchmark_id",
    "benchmark_kind",
    "chunk_id",
    "profile_id",
    "selection_seed",
    "role",
]
SAVED_TABLES = [
    "accuracy_vs_proxy.csv",
    "results.csv",
    "allocations.csv",
    "reference_allocations.csv",
    "benchmark_optima.json",
    "validation_report.json",
]


def versions():
    result = {"python": platform.python_version()}
    for name in ("gurobipy", "pandas", "plotly", "PyYAML"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    # Source identity prevents stale certification after mathematical implementation changes.
    result["solver_implementation"] = digest(
        {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("optimization.py", "experiment_solver.py", "proxy_solver.py")
        }
    )
    return result


def run_identity(resolved):
    value = dict(resolved)
    value["models"] = sorted(value["models"])
    return digest(value)


def allocation_rows(chosen, solution_id, mid, instance, seed, role, benchmark):
    return [
        {
            "solution_id": solution_id,
            "patient_id": p,
            "clinician_id": chosen[p],
            "model_id": mid,
            "benchmark_id": benchmark,
            "benchmark_kind": "proxy",
            "chunk_id": instance["chunk_id"],
            "profile_id": instance["profile_id"],
            "selection_seed": seed,
            "role": role,
        }
        for p in sorted(chosen)
    ]


def run(args):
    prepared = prepare(args)
    resolved = prepared["resolved"]
    tolerance = number(args.optimality_tolerance, "optimality-tolerance", 1e-9)
    if tolerance > 1e-3:
        raise ValueError("optimality-tolerance must be <= 0.001")
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
        versions=versions(),
        solver_parameters=settings.parameters(),
        intended_objective_degradation=0,
        random_algorithm="SHA256 instance/seed/purpose; sorted IDs; MT19937 iid U(0,1)",
        selection_warning="Random linear objectives do not sample optimal assignments uniformly",
    )
    seeds = resolved["selection_seeds"]
    all_seeds = list(dict.fromkeys([*seeds, args.chunk_seed]))
    feasible = sum(i["feasible"] for i in prepared["instances"])
    planned = {
        "nominal_stages": feasible
        * (len(args.models) + 1)
        * (2 if args.objective == "lexicographic" else 1),
        "benchmark_selections": feasible,
        "candidate_selections": feasible * len(args.models) * len(all_seeds),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "benchmark_id": args.benchmark,
                    "benchmark_kind": "proxy",
                    "n_patients": sum(map(len, resolved["chunks"].values())),
                    "included_instances": feasible,
                    "infeasible_instances": [i for i in prepared["instances"] if not i["feasible"]],
                    "expected_solve_counts_before_cache": planned,
                },
                indent=2,
            )
        )
        return 0
    identity = run_identity(resolved)
    output = args.output.resolve()
    if args.resume:
        path = output / "resolved_config.json"
        if not path.is_file():
            raise ValueError("--resume requires an existing resolved_config.json")
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_fingerprint = previous.pop("run_fingerprint", None)
        if previous_fingerprint != identity or run_identity(previous) != identity:
            raise ValueError(
                "Incompatible resume: inputs, membership, profiles, seeds or solver settings changed"
            )
    else:
        output = fresh_directory(output)
    resolved["run_fingerprint"] = identity
    write_json(output / "resolved_config.json", resolved)
    cache = ArtifactCache(args.cache_dir or output / "cache")
    rows, allocations, references, bench_records, failures = [], [], [], [], []
    # Start Gurobi only if an uncached solve is needed, including on resume.
    with ExitStack() as stack:
        active = None

        def session():
            nonlocal active
            if active is None:
                active = stack.enter_context(SolverSession(settings))
            return active

        def nominal(instance, mid):
            data = prepared["loaded"][mid]
            weights = {p: data["weights"][p] for p in instance["patients"]}
            key = {
                "kind": "proxy_nominal_v1",
                "instance": canonical(instance),
                "weights": weights,
                "input_sha256": data["sha256"],
                "objective": args.objective,
                "solver": settings.parameters(),
                "tolerance": tolerance,
                "versions": resolved["versions"],
            }
            optimum = cache.get(
                key,
                lambda: solver.nominal(
                    session(), canonical(instance), weights, args.objective, tolerance
                ),
            )
            if not optimum.get("certified") or not all(
                s.get("certified") for s in optimum["stages"]
            ):
                raise ValueError("Uncertified cached nominal solve")
            solver.verify(instance, weights, optimum["assignment"], optimum, tolerance)
            return key, optimum, weights

        def selection(instance, key, optimum, weights, seed, purpose):
            selection_key = {
                "kind": "proxy_selection_v1",
                "nominal": key,
                "seed": seed,
                "purpose": purpose,
            }
            result = cache.get(
                selection_key,
                lambda: select(session(), instance, weights, optimum, tolerance, seed, purpose),
            )
            if not result.get("certified") or not all(s.get("certified") for s in result["stages"]):
                raise ValueError("Uncertified cached selection")
            solver.verify(instance, weights, result["assignment"], optimum, tolerance)
            return digest(selection_key), result

        for instance in prepared["instances"]:
            context = {k: instance[k] for k in ("chunk_id", "profile_id")}
            bench = dict(benchmark_id=args.benchmark, benchmark_kind="proxy", **context)
            reference_id, reference, bo, bw = "", None, None, None
            benchmark_error = None
            try:
                if not instance["feasible"]:
                    raise ValueError(
                        "INFEASIBLE: chunk exceeds remaining clinician count capacities"
                    )
                bk, bo, bw = nominal(instance, args.benchmark)
                reference_id, reference = selection(
                    instance, bk, bo, bw, args.benchmark_seed, "benchmark"
                )
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
                bench.update(
                    proxy_optimum=bo["g"],
                    count_optimum=bo["d"],
                    solver_status="OPTIMAL",
                    nominal_stages=bo["stages"],
                    reference_stages=reference["stages"],
                    reference_solution_id=reference_id,
                    reference_seed=args.benchmark_seed,
                )
            except (ValueError, RuntimeError) as exc:
                benchmark_error = str(exc)
                bench.update(
                    solver_status="FAILED",
                    failure_reason=benchmark_error,
                    proxy_optimum=bo["g"] if bo else None,
                    diagnostics=getattr(exc, "diagnostics", None),
                )
            bench_records.append(bench)
            for mid in args.models:
                nominal_error, optimum, key, weights = benchmark_error, None, None, None
                if not benchmark_error:
                    try:
                        key, optimum, weights = nominal(instance, mid)
                    except (ValueError, RuntimeError) as exc:
                        nominal_error = str(exc)
                for seed in all_seeds:
                    row = dict(
                        model_id=mid,
                        benchmark_id=args.benchmark,
                        benchmark_kind="proxy",
                        objective=args.objective,
                        **context,
                        n_patients=len(instance["patients"]),
                        selection_seed=seed,
                        sample_index=seeds.index(seed) if seed in seeds else None,
                        reference_solution_id=reference_id,
                        reference_seed=args.benchmark_seed,
                        is_seed_sample=seed in seeds,
                        is_chunk_evaluation=seed == args.chunk_seed,
                        nominal_optimum=optimum["g"] if optimum else None,
                        proxy_optimum=bo["g"] if bo else None,
                    )
                    try:
                        if nominal_error:
                            raise ValueError(nominal_error)
                        sid, selected = selection(
                            instance, key, optimum, weights, seed, "candidate"
                        )
                        chosen = selected["assignment"]
                        metrics = evaluate(instance, bw, bo, chosen, reference["assignment"])
                        if metrics["proxy_regret"] < -tolerance:
                            raise ValueError(
                                "Proxy evaluation is below the certified benchmark optimum"
                            )
                        row.update(
                            solution_id=sid,
                            solution_hash=digest(chosen),
                            **metrics,
                            solver_status="OPTIMAL",
                            optimality_gap=selected["stages"][-1]["gap"],
                        )
                        role = (
                            "seed_sample+chunk_evaluation"
                            if seed in seeds and seed == args.chunk_seed
                            else ("seed_sample" if seed in seeds else "chunk_evaluation")
                        )
                        allocations.extend(
                            allocation_rows(chosen, sid, mid, instance, seed, role, args.benchmark)
                        )
                    except (ValueError, RuntimeError) as exc:
                        row.update(solver_status="FAILED", failure_reason=str(exc))
                        failures.append(
                            dict(
                                model_id=mid,
                                **context,
                                selection_seed=seed,
                                reason=str(exc),
                                diagnostics=getattr(exc, "diagnostics", None),
                            )
                        )
                    rows.append(row)
    distinct = []
    for instance in prepared["instances"]:
        for mid in args.models:
            sample = [
                r
                for r in rows
                if r["model_id"] == mid
                and r["chunk_id"] == instance["chunk_id"]
                and r["profile_id"] == instance["profile_id"]
                and r["is_seed_sample"]
            ]
            distinct.append(
                {
                    "model_id": mid,
                    "chunk_id": instance["chunk_id"],
                    "profile_id": instance["profile_id"],
                    "expected_samples": len(seeds),
                    "successful_samples": sum(r["solver_status"] == "OPTIMAL" for r in sample),
                    "distinct_allocations": len(
                        {r["solution_hash"] for r in sample if r["solver_status"] == "OPTIMAL"}
                    ),
                }
            )
    report = {
        "benchmark_id": args.benchmark,
        "benchmark_kind": "proxy",
        "planned_solve_counts": planned,
        "expected_allocations": len(rows),
        "successful_allocations": len(rows) - len(failures),
        "failed_allocations": len(failures),
        "total_instances": len(prepared["instances"]),
        "feasible_instances": feasible,
        "infeasible_instances": len(prepared["instances"]) - feasible,
        "complete_model_instances": sum(
            d["successful_samples"] == len(seeds)
            and all(
                r["solver_status"] == "OPTIMAL"
                for r in rows
                if all(r[k] == d[k] for k in ("model_id", "chunk_id", "profile_id"))
            )
            for d in distinct
        ),
        "failures": failures,
        "distinct_allocations": distinct,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }
    write_csv(output / "accuracy_vs_proxy.csv", prepared["accuracy"])
    write_csv(output / "results.csv", rows, RESULT_COLUMNS)
    write_csv(output / "allocations.csv", allocations, ALLOCATION_COLUMNS)
    write_csv(output / "reference_allocations.csv", references, ALLOCATION_COLUMNS)
    write_json(output / "benchmark_optima.json", bench_records)
    write_json(output / "validation_report.json", report)
    write_json(
        output / "artifacts.json",
        {name: file_hash(output / name) for name in [*SAVED_TABLES, "resolved_config.json"]},
    )
    from .proxy_plots import render

    render(output, output, args.show_type, args.facet_cols, allow_existing=True)
    print(
        f"Proxy benchmark {args.benchmark}: {report['successful_allocations']}/{len(rows)} "
        f"certified allocations; report: {output / 'index.html'}"
    )
    return 0 if not failures else 2
