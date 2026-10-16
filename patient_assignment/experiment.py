"""Reproducible experiment runner. Never imports an inference engine."""

from __future__ import annotations

import json
import platform
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .experiment_inputs import digest, scenario_set, seed_for
from .inputs import file_hash
from .reporting import fresh_directory, write_csv, write_json


def versions():
    result = {"python": platform.python_version()}
    for package in ("gurobipy", "pandas", "PyYAML", "matplotlib", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    result["implementation"] = digest(
        {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )
    return result


class ArtifactCache:
    """Content-addressed, checksummed completed components; failures are never cached."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, identity, compute):
        key = digest(identity)
        path = self.path / (key + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("identity") != identity or saved.get("payload_hash") != digest(
                saved.get("payload")
            ):
                raise ValueError(f"Stale/corrupt cache record: {path}")
            self.hits += 1
            return saved["payload"]
        self.misses += 1
        payload = compute()
        record = {"identity": identity, "payload": payload, "payload_hash": digest(payload)}
        # Exclusive creation preserves historical components, even across concurrent runs.
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True, allow_nan=False)
        except FileExistsError:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("identity") != identity or saved.get("payload_hash") != digest(
                saved.get("payload")
            ):
                raise ValueError(f"Concurrent cache record is invalid: {path}")
            return saved["payload"]
        return payload


def preview(prepared):
    o = prepared["options"]
    count = {
        "nominal": 0,
        "random_selection": 0,
        "benchmark_selection": 0,
        "matching": 0,
        "robust": 0,
        "regret_bounds": 0,
    }
    nmodels = len(o["models"])
    ninputs = len(prepared["inputs"])
    for instance in prepared["instances"]:
        if not instance["feasible"]:
            continue
        for mode in o["objectives"]:
            count["nominal"] += ninputs * (2 if mode == "lexicographic" else 1)
            if "random" in o["selection"]:
                count["random_selection"] += nmodels * o["samples"]
            pairing = max(
                o["samples"] if "random" in o["selection"] else 0,
                o["robust_samples"] if "robust" in o["selection"] else 0,
            )
            count["benchmark_selection"] += pairing
            count["matching"] += nmodels * int("matched" in o["selection"])
            count["robust"] += nmodels * (1 + o["robust_samples"]) * int("robust" in o["selection"])
            j = len(instance["clinicians"])
            count["regret_bounds"] += (
                nmodels * (1 + j * (j - 1)) * int(o["regret_bounds"] == "exact")
            )
    return {
        "models": o["models"],
        "benchmark": prepared["benchmark"],
        "chunks": prepared["chunks"],
        "profiles": prepared["profiles"],
        "instances": prepared["instances"],
        "exclusions": prepared["exclusions"],
        "missing_uncertainty": prepared["missing_uncertainty"],
        "planned_solve_stages_before_cache": count,
        "warning": "Synthetic profiles/scenarios are experimental conditions; no inference is run.",
    }


def summarize(rows, instances):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    group = [
        "model_id",
        "objective",
        "profile_id",
        "strategy",
        "oracle",
        "stress_set_id",
        "heldout_set_id",
    ]
    metrics = [
        "regret",
        "benchmark_range",
        "nominal_range",
        "count_range",
        "disagreement",
        "stress_w",
        "heldout_w",
    ]
    per_instance = []
    for key, part in frame.groupby([*group, "chunk_id"], dropna=False, sort=True):
        valid = part.loc[part.certified.eq(True)]
        row = {
            **dict(zip([*group, "chunk_id"], key)),
            "level": "instance",
            "draws": len(part),
            "valid": len(valid),
            "failed": len(part) - len(valid),
            "unique_allocations": valid.assignment_hash.nunique()
            if "assignment_hash" in valid
            else 0,
            "patient_count": int(part.patient_count.iloc[0]),
        }
        for metric in metrics:
            values = pd.to_numeric(
                valid.get(metric, pd.Series(dtype=float)), errors="coerce"
            ).dropna()
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_min"] = float(values.min()) if len(values) else None
            row[metric + "_max"] = float(values.max()) if len(values) else None
        per_instance.append(row)
    # Exclude chunk-specific stress-set IDs from regret aggregation; never pool stress W.
    aggregate_keys = ["model_id", "objective", "profile_id", "strategy", "oracle", "patient_count"]
    result = list(per_instance)
    for key, part in pd.DataFrame(per_instance).groupby(aggregate_keys, dropna=False, sort=True):
        row = {
            **dict(zip(aggregate_keys, key)),
            "level": "across_chunks",
            "instance_count": len(part),
            "valid_instances": int((part.valid > 0).sum()),
            "failed_instances": int((part.valid == 0).sum()),
            "draws": int(part.draws.sum()),
            "valid": int(part.valid.sum()),
            "failed": int(part.failed.sum()),
            "unique_allocations": int(part.unique_allocations.sum()),
            "weighting": "equal chunks, stratified by effective patient count; profiles separate",
            "excluded_infeasible_instances": sum(not i["feasible"] for i in instances),
        }
        for metric in metrics:
            if metric in {"stress_w", "heldout_w"}:
                continue
            values = pd.to_numeric(part[metric + "_mean"], errors="coerce").dropna()
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
        result.append(row)
    # csv must not contain opaque NaN IDs introduced by groupby.
    return [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        for r in result
    ]


def agreement(prepared, mid, patients):
    a, b = prepared["inputs"][mid], prepared["inputs"][prepared["benchmark"]["id"]]
    label = (
        "agreement with proxy benchmark"
        if prepared["benchmark"]["kind"] == "proxy"
        else "agreement with clinical reference"
    )
    if a["entry"]["value_type"] != "category" or b["entry"]["value_type"] != "category":
        return {"category_agreement_label": label, "category_disagreement": None, "qwk": None}
    labels = sorted(
        prepared["scoring"]["category_weights"], key=prepared["scoring"]["category_weights"].get
    )
    aa, bb = [a["categories"][p] for p in patients], [b["categories"][p] for p in patients]
    from sklearn.metrics import cohen_kappa_score

    # Constant identical ratings have undefined kappa, rather than a fabricated perfect score.
    qwk = (
        None
        if len(set(aa + bb)) == 1
        else float(cohen_kappa_score(aa, bb, labels=labels, weights="quadratic"))
    )
    return {
        "category_agreement_label": label,
        "category_disagreement": sum(x != y for x, y in zip(aa, bb)) / len(patients),
        "qwk": qwk,
    }


def run_experiment(prepared):
    o = prepared["options"]
    if o["dry_run"]:
        print(json.dumps(preview(prepared), indent=2, allow_nan=False))
        return 0
    if not o["output"]:
        raise ValueError("Supply --output or output in YAML")
    output = Path(o["output"])
    package_versions = versions()
    from .experiment_solver import ExperimentSettings

    settings = ExperimentSettings(o["time_limit"], o["threads"], o["solver_seed"], o["mip_gap"])
    vectors = {}
    for instance in prepared["instances"]:
        if instance["feasible"]:
            for mid in o["models"]:
                key = digest([instance["chunk_id"], instance["profile_id"], mid])
                vectors[key] = {
                    "chunk_id": instance["chunk_id"],
                    "profile_id": instance["profile_id"],
                    "model_id": mid,
                    "construction": scenario_set(prepared, mid, instance),
                    "heldout": scenario_set(prepared, mid, instance, heldout=True),
                }
    manifest = {
        **prepared,
        "versions": package_versions,
        "solver_parameters": settings.parameters(),
        "uncertainty_vectors": vectors,
        "interpretation": {
            "regret": "proxy-benchmark regret"
            if prepared["benchmark"]["kind"] == "proxy"
            else "clinical-reference regret",
            "sampling": "independent Uniform(0,1) linear objectives; not uniform optimal-set sampling",
            "stress": "in-set performance unless heldout_w is present; no calibrated probabilities",
            "profiles": "synthetic unless metadata documents observed operational provenance",
        },
    }
    fingerprint_payload = {
        **manifest,
        "options": {
            k: v for k, v in o.items() if k not in {"resume", "dry_run", "output", "cache_dir"}
        },
    }
    manifest["run_fingerprint"] = digest(fingerprint_payload)
    if o["resume"] and (output / "resolved_config.json").exists():
        previous = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != manifest["run_fingerprint"]:
            raise ValueError("Resume fingerprint mismatch; choose a new output directory")
        previous_payload = {k: v for k, v in previous.items() if k != "run_fingerprint"}
        previous_payload["options"] = {
            k: v
            for k, v in previous["options"].items()
            if k not in {"resume", "dry_run", "output", "cache_dir"}
        }
        if digest(previous_payload) != previous["run_fingerprint"]:
            raise ValueError("Resume manifest checksum mismatch")
        status_path = output / "run_status.json"
        if status_path.exists():
            previous_status = json.loads(status_path.read_text(encoding="utf-8"))
            if previous_status.get("complete"):
                for name, expected in previous_status["artifacts"].items():
                    if not (output / name).is_file() or file_hash(output / name) != expected:
                        raise ValueError(f"Resume artifact checksum mismatch: {name}")
                print(f"Reused complete matching run: {output}")
                return 0
        # Immutable configuration remains in place; only derived incomplete tables are regenerated.
    else:
        fresh_directory(output)
        write_json(output / "resolved_config.json", manifest)
    write_json(
        output / "run_status.json",
        {"complete": False, "state": "running", "run_fingerprint": manifest["run_fingerprint"]},
    )
    write_csv(
        output / "instances.csv",
        [
            {**i, "patients": json.dumps(i["patients"]), "clinicians": json.dumps(i["clinicians"])}
            for i in prepared["instances"]
        ],
    )
    write_csv(
        output / "effective_membership.csv",
        [
            {"chunk_id": chunk, "patient_id": p}
            for chunk, ids in prepared["chunks"].items()
            for p in ids
        ],
        ["chunk_id", "patient_id"],
    )
    write_csv(
        output / "exclusions.csv", prepared["exclusions"], ["chunk_id", "patient_id", "reason"]
    )
    write_json(output / "uncertainty_manifest.json", vectors)
    write_csv(
        output / "uncertainty_scores.csv",
        [
            {
                "chunk_id": v["chunk_id"],
                "profile_id": v["profile_id"],
                "model_id": v["model_id"],
                "set": kind,
                "vector_id": s["vector_id"],
                "multiplicity": s["multiplicity"],
                "patient_id": p,
                "score": w,
            }
            for v in vectors.values()
            for kind in ("construction", "heldout")
            for s in v[kind]
            for p, w in s["weights"].items()
        ],
        [
            "chunk_id",
            "profile_id",
            "model_id",
            "set",
            "vector_id",
            "multiplicity",
            "patient_id",
            "score",
        ],
    )
    from .experiment_solver import (
        IncompleteSolve,
        actual,
        matched_selection,
        nominal,
        random_selection,
        regret_envelope,
        robust_selection,
        stress,
    )
    from .optimization import SolverDependencyError, SolverSession

    cache = ArtifactCache(o["cache_dir"] or output.parent / ".assignment_cache")
    rows, assignments, loads, bounds, failures, benchmark_draws, witness_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tolerance = o["optimality_tolerance"]
    bench_id = prepared["benchmark"]["id"]
    benchmark_weights = prepared["inputs"][bench_id]["weights"]

    def failure(context, exc):
        failures.append(
            {**context, "error": str(exc), "diagnostics": getattr(exc, "diagnostics", None)}
        )

    def save(chosen, paired, context, optimum, bench_optimum, construction, heldout, selected):
        mid = context["model_id"]
        weights = prepared["inputs"][mid]["weights"]
        nominal_values, reference_values = (
            actual(instance, weights, chosen),
            actual(instance, benchmark_weights, chosen),
        )
        raw = reference_values["range"] - bench_optimum["g"]
        if raw < -tolerance - 1e-8:
            raise ValueError(
                f"Substantial negative regret: {raw}; domains or certification are inconsistent"
            )
        sid = digest([manifest["run_fingerprint"], context])[:24]
        stages = selected["stages"]
        row = {
            **context,
            "solution_id": sid,
            "certified": True,
            "status": stages[-1]["status"],
            "assignment_hash": digest(chosen),
            "benchmark_id": bench_id,
            "benchmark_kind": prepared["benchmark"]["kind"],
            "regret_label": manifest["interpretation"]["regret"],
            "raw_regret": raw,
            "regret": max(0, raw),
            "benchmark_range": reference_values["range"],
            "benchmark_optimum": bench_optimum["g"],
            "nominal_optimum": optimum["g"],
            "nominal_range": nominal_values["range"],
            "count_optimum": optimum["d"],
            "count_range": nominal_values["count_range"],
            "disagreement": sum(chosen[p] != paired[p] for p in instance["patients"])
            / len(instance["patients"]),
            "stress_w": stress(instance, construction, chosen),
            "heldout_w": stress(instance, heldout, chosen),
            "robust_optimum": selected.get("robust_optimum"),
            "seed": selected.get("seed"),
            "nominal_slack": o["nominal_slack"] if context["strategy"] == "robust" else 0,
            "selection_label": ("near-optimal robust" if o["nominal_slack"] else "robust tie-break")
            if context["strategy"] == "robust"
            else context["strategy"],
            "selection_runtime": sum(s["runtime"] for s in stages),
            "selection_objective": stages[-1]["objective"],
            "selection_bound": stages[-1]["bound"],
            "selection_gap": stages[-1]["gap"],
            "optimality_tolerance": tolerance,
            "stages": json.dumps(stages),
            "nominal_stages": json.dumps(optimum["stages"]),
            **agreement(prepared, mid, instance["patients"]),
        }
        if context["oracle"]:
            row["selection_label"] = "ORACLE diagnostic: " + row["selection_label"]
        pending_loads = []
        for own, ref in zip(nominal_values["loads"], reference_values["loads"]):
            pending_loads.append(
                {"solution_id": sid, **own, "benchmark_load": ref["final_severity"]}
            )
        rows.append(row)
        loads.extend(pending_loads)
        assignments.extend(
            {
                "solution_id": sid,
                "patient_id": p,
                "clinician_id": chosen[p],
                "candidate_score": weights[p],
                "benchmark_score": benchmark_weights[p],
                "benchmark_clinician_id": paired[p],
            }
            for p in instance["patients"]
        )

    try:
        with SolverSession(settings) as session:
            for instance in prepared["instances"]:
                if not instance["feasible"]:
                    continue
                for mode in o["objectives"]:
                    identity_base = {
                        "versions": package_versions,
                        "patients": instance["patients"],
                        "chunk": instance["chunk_id"],
                        "profile": instance["profile_id"],
                        "clinicians": instance["clinicians"],
                        "objective": mode,
                        "solver": settings.parameters(),
                        "tolerance": tolerance,
                    }
                    optima, nominal_ids = {}, {}
                    for mid, source in prepared["inputs"].items():
                        identity = {
                            **identity_base,
                            "task": "nominal",
                            "input_id": mid,
                            "content_id": source["content_id"],
                        }
                        nominal_ids[mid] = digest(identity)
                        try:
                            optima[mid] = cache.get(
                                identity,
                                partial(
                                    nominal, session, instance, source["weights"], mode, tolerance
                                ),
                            )
                        except (IncompleteSolve, ValueError, session.gp.GurobiError) as exc:
                            failure(
                                {
                                    "chunk_id": instance["chunk_id"],
                                    "profile_id": instance["profile_id"],
                                    "model_id": mid,
                                    "objective": mode,
                                    "stage": "nominal",
                                },
                                exc,
                            )
                    paired_draws = {}
                    pairing_count = max(
                        o["samples"] if "random" in o["selection"] else 0,
                        o["robust_samples"] if "robust" in o["selection"] else 0,
                    )
                    if bench_id in optima:
                        for index in range(pairing_count):
                            seed = seed_for(
                                o["sample_seed"],
                                instance["chunk_id"],
                                instance["profile_id"],
                                mode,
                                bench_id,
                                prepared["inputs"][bench_id]["content_id"],
                                "benchmark",
                                index,
                            )
                            try:
                                selected = cache.get(
                                    {
                                        "task": "benchmark_random",
                                        "nominal": nominal_ids[bench_id],
                                        "seed": seed,
                                    },
                                    partial(
                                        random_selection,
                                        session,
                                        instance,
                                        benchmark_weights,
                                        optima[bench_id],
                                        tolerance,
                                        seed,
                                    ),
                                )
                                paired_draws[index] = selected["assignment"]
                                benchmark_draws.append(
                                    {
                                        "chunk_id": instance["chunk_id"],
                                        "profile_id": instance["profile_id"],
                                        "objective": mode,
                                        "sample": index,
                                        "benchmark_id": bench_id,
                                        **selected,
                                    }
                                )
                            except (IncompleteSolve, ValueError, session.gp.GurobiError) as exc:
                                failure(
                                    {
                                        "stage": "benchmark selection",
                                        "chunk_id": instance["chunk_id"],
                                        "objective": mode,
                                        "sample": index,
                                    },
                                    exc,
                                )
                    for mid in o["models"]:
                        source = prepared["inputs"][mid]
                        set_key = digest([instance["chunk_id"], instance["profile_id"], mid])
                        construction, heldout = (
                            vectors[set_key]["construction"],
                            vectors[set_key]["heldout"],
                        )
                        stress_id = digest([s["weights"] for s in construction])
                        heldout_id = digest([s["weights"] for s in heldout]) if heldout else "none"
                        common = {
                            "model_id": mid,
                            "benchmark_id": bench_id,
                            "benchmark_kind": prepared["benchmark"]["kind"],
                            "regret_label": manifest["interpretation"]["regret"],
                            "chunk_id": instance["chunk_id"],
                            "profile_id": instance["profile_id"],
                            "objective": mode,
                            "patient_count": len(instance["patients"]),
                            "oracle": mid in prepared["oracle_models"],
                            "stress_set_id": stress_id,
                            "heldout_set_id": heldout_id,
                        }
                        if o["regret_bounds"] == "exact" and mid in optima and bench_id in optima:
                            try:
                                envelope = regret_envelope(
                                    session,
                                    instance,
                                    source["weights"],
                                    benchmark_weights,
                                    optima[mid],
                                    optima[bench_id],
                                    tolerance,
                                )
                                bound_row = {
                                    **common,
                                    **{
                                        k: v
                                        for k, v in envelope.items()
                                        if k not in {"witnesses", "statuses"}
                                    },
                                    "statuses": json.dumps(envelope["statuses"]),
                                    "restriction": "original nominal-optimal set; zero intentional slack",
                                    "optimality_tolerance": tolerance,
                                }
                                for extremum, witness in envelope["witnesses"].items():
                                    wid = digest([common, extremum, witness])[:24]
                                    bound_row[extremum + "_witness_id"] = wid
                                    witness_rows.extend(
                                        {"witness_id": wid, "patient_id": p, "clinician_id": j}
                                        for p, j in witness.items()
                                    )
                                bounds.append(bound_row)
                                if not envelope["complete"]:
                                    failure(
                                        {**common, "stage": "regret envelope"},
                                        ValueError(
                                            "Incomplete exact envelope; inspect solver bounds"
                                        ),
                                    )
                            except (ValueError, session.gp.GurobiError) as exc:
                                bounds.append({**common, "complete": False, "error": str(exc)})
                                failure({**common, "stage": "regret envelope"}, exc)
                        elif o["regret_bounds"] == "exact":
                            bounds.append(
                                {
                                    **common,
                                    "complete": False,
                                    "error": "Required original optimum uncertified; envelope not attempted",
                                }
                            )
                        robust_results, robust_error = None, None
                        for strategy in o["selection"]:
                            count = (
                                o["samples"]
                                if strategy == "random"
                                else (o["robust_samples"] if strategy == "robust" else 1)
                            )
                            for index in range(count):
                                context = {**common, "strategy": strategy, "sample": index}
                                try:
                                    if mid not in optima or bench_id not in optima:
                                        raise ValueError(
                                            "Required nominal optimum uncertified; selection not attempted"
                                        )
                                    optimum, bench_optimum = optima[mid], optima[bench_id]
                                    seed = seed_for(
                                        o["sample_seed"],
                                        instance["chunk_id"],
                                        instance["profile_id"],
                                        mode,
                                        mid,
                                        source["content_id"],
                                        strategy,
                                        index,
                                    )
                                    if strategy == "random":
                                        selected = cache.get(
                                            {
                                                "task": "random",
                                                "nominal": nominal_ids[mid],
                                                "seed": seed,
                                            },
                                            partial(
                                                random_selection,
                                                session,
                                                instance,
                                                source["weights"],
                                                optimum,
                                                tolerance,
                                                seed,
                                            ),
                                        )
                                    elif strategy == "matched":
                                        selected = cache.get(
                                            {
                                                "task": "matched",
                                                "nominal": nominal_ids[mid],
                                                "benchmark": nominal_ids[bench_id],
                                            },
                                            partial(
                                                matched_selection,
                                                session,
                                                instance,
                                                source["weights"],
                                                benchmark_weights,
                                                optimum,
                                                bench_optimum,
                                                tolerance,
                                            ),
                                        )
                                    else:
                                        if robust_error is not None:
                                            raise robust_error
                                        if robust_results is None:
                                            seeds = [
                                                seed_for(
                                                    o["sample_seed"],
                                                    instance["chunk_id"],
                                                    instance["profile_id"],
                                                    mode,
                                                    mid,
                                                    source["content_id"],
                                                    strategy,
                                                    r,
                                                )
                                                for r in range(o["robust_samples"])
                                            ]
                                            try:
                                                robust_results = cache.get(
                                                    {
                                                        "task": "robust",
                                                        "nominal": nominal_ids[mid],
                                                        "stress": stress_id,
                                                        "slack": o["nominal_slack"],
                                                        "seeds": seeds,
                                                    },
                                                    partial(
                                                        robust_selection,
                                                        session,
                                                        instance,
                                                        source["weights"],
                                                        optimum,
                                                        construction,
                                                        tolerance,
                                                        o["nominal_slack"],
                                                        seeds,
                                                    ),
                                                )
                                            except (ValueError, session.gp.GurobiError) as exc:
                                                robust_error = exc
                                                raise
                                        selected = robust_results[index]
                                    paired = selected.get(
                                        "benchmark_assignment", paired_draws.get(index)
                                    )
                                    if paired is None:
                                        raise ValueError(
                                            "Independent benchmark sample uncertified; no paired metrics exported"
                                        )
                                    save(
                                        selected["assignment"],
                                        paired,
                                        context,
                                        optimum,
                                        bench_optimum,
                                        construction,
                                        heldout,
                                        selected,
                                    )
                                except (ValueError, session.gp.GurobiError) as exc:
                                    failure(context, exc)
                                    rows.append({**context, "certified": False, "error": str(exc)})
                print(f"{instance['chunk_id']} / {instance['profile_id']}: processed", flush=True)
    except SolverDependencyError as exc:
        failure({"stage": "solver initialization"}, exc)
    write_csv(
        output / "solutions.csv",
        rows,
        [
            "solution_id",
            "model_id",
            "chunk_id",
            "profile_id",
            "objective",
            "strategy",
            "sample",
            "certified",
            "assignment_hash",
            "regret",
            "benchmark_range",
            "count_range",
            "disagreement",
            "stress_w",
            "heldout_w",
        ],
    )
    write_csv(
        output / "assignments.csv",
        assignments,
        ["solution_id", "patient_id", "clinician_id", "candidate_score", "benchmark_score"],
    )
    write_csv(
        output / "clinician_loads.csv",
        loads,
        ["solution_id", "clinician_id", "final_count", "final_severity", "benchmark_load"],
    )
    write_csv(
        output / "regret_bounds.csv",
        bounds,
        [
            "model_id",
            "chunk_id",
            "profile_id",
            "objective",
            "complete",
            "minimum_regret",
            "maximum_regret",
        ],
    )
    write_csv(
        output / "regret_witnesses.csv", witness_rows, ["witness_id", "patient_id", "clinician_id"]
    )
    write_json(output / "benchmark_samples.json", benchmark_draws)
    write_json(
        output / "validation_report.json",
        {
            "failures": failures,
            "excluded_instances": [i for i in prepared["instances"] if not i["feasible"]],
        },
    )
    summary_rows = summarize(rows, prepared["instances"])
    for row in summary_rows:
        row.update(
            benchmark_id=bench_id,
            benchmark_kind=prepared["benchmark"]["kind"],
            regret_label=manifest["interpretation"]["regret"],
        )
    write_csv(
        output / "summaries.csv",
        summary_rows,
        ["level", "model_id", "objective", "profile_id", "strategy", "draws", "valid", "failed"],
    )
    write_csv(
        output / "deployable_summaries.csv",
        [r for r in summary_rows if not r.get("oracle") and r["strategy"] != "matched"],
        ["level", "model_id", "objective", "profile_id", "strategy"],
    )
    complete = not failures and all(i["feasible"] for i in prepared["instances"])
    artifacts = {
        p.name: file_hash(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "run_status.json"
    }
    write_json(
        output / "run_status.json",
        {
            "complete": complete,
            "failure_count": len(failures),
            "excluded_instances": sum(not i["feasible"] for i in prepared["instances"]),
            "valid_selections": sum(bool(r["certified"]) for r in rows),
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "run_fingerprint": manifest["run_fingerprint"],
            "artifacts": artifacts,
        },
    )
    print(f"{'Complete' if complete else 'Incomplete (see validation_report.json)'}: {output}")
    return 0 if complete else 1
