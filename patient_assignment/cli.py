"""New commands isolated from model inference and legacy classification reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunking import create_chunks
from .cohort import ID_COLUMN, add_filter_arguments
from .inputs import (
    check_note_hashes,
    load_membership,
    load_severity,
    parse_clinicians,
    preflight_capacity,
)
from .optimization import (
    SolverDependencyError,
    SolverSession,
    SolverSettings,
    evaluate,
    solve_lexicographic,
    verify_optimal_set,
)
from .reporting import (
    base_config,
    comparison_summary,
    flatten_stages,
    flatten_status,
    fresh_directory,
    write_csv,
    write_json,
)
from .selection import (
    RANDOM_ALGORITHM,
    coefficients,
    match_optima,
    random_optimum,
    reference_regret,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Independent patient chunks and decision allocation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    chunk = commands.add_parser(
        "chunk", help="Create reusable membership without notes/inference/solver"
    )
    chunk.add_argument("--input", type=Path, required=True)
    chunk.add_argument("--sheet")
    chunk.add_argument("--id-column", default=ID_COLUMN)
    add_filter_arguments(chunk)
    chunk.add_argument("--chunk-size", type=int, required=True)
    chunk.add_argument("--seed", type=int, default=42)
    chunk.add_argument("--output", type=Path, required=True)
    for command in ("solve", "compare-assignments"):
        sub = commands.add_parser(command)
        sub.add_argument("--chunks", type=Path, required=True)
        sub.add_argument(
            "--clinician",
            nargs=3,
            action="append",
            required=True,
            metavar=("NAME", "INITIAL_COUNT", "INITIAL_SEVERITY"),
        )
        sub.add_argument("--clinician-capacity", type=int, required=True)
        sub.add_argument(
            "--time-limit",
            type=float,
            help="Positive seconds per optimization stage; default unlimited",
        )
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--solver-seed", type=int, default=0)
        sub.add_argument("--output", type=Path, required=True)
        suffixes = ("",) if command == "solve" else ("-a", "-b")
        for suffix in suffixes:
            sub.add_argument("--predictions" + suffix, type=Path, required=True)
            sub.add_argument("--label" + suffix, required=True)
            sub.add_argument("--id-column" + suffix, default=ID_COLUMN)
            sub.add_argument("--category-column" + suffix, default="treatment_category")
            sub.add_argument("--sheet" + suffix)
        if command == "compare-assignments":
            sub.add_argument("--selection-mode", choices=("matched", "random"), default="matched")
            sub.add_argument("--selection-seed", type=int, default=42)
            sub.add_argument("--selection-repeats", type=int, default=1)
            sub.add_argument("--reference-side", choices=("A", "B"))
    return parser


def preflight(args):
    settings = SolverSettings(args.time_limit, args.threads, args.solver_seed)
    settings.validate()
    clinicians = parse_clinicians(args.clinician, args.clinician_capacity)
    chunks = load_membership(args.chunks)
    preflight_capacity(chunks, clinicians, args.clinician_capacity)
    patients = [p for ids in chunks.values() for p in ids]
    labels, predictions = {}, {}
    for side in ("",) if args.command == "solve" else ("a", "b"):
        suffix = "_" + side if side else ""
        label = getattr(args, "label" + suffix).strip()
        if not label or label in labels.values():
            raise ValueError("Model labels must be nonblank and unique")
        labels[side] = label
        predictions[side] = load_severity(
            getattr(args, "predictions" + suffix),
            patients,
            getattr(args, "id_column" + suffix),
            getattr(args, "category_column" + suffix),
            getattr(args, "sheet" + suffix),
        )
    notes = None
    if args.command == "compare-assignments":
        if args.selection_seed < 0 or args.selection_repeats < 1:
            raise ValueError("Selection seed must be nonnegative; repeats must be positive")
        if args.selection_mode == "matched" and args.selection_repeats != 1:
            raise ValueError("Selection repeats greater than one are random-mode only")
        notes = check_note_hashes(predictions["a"], predictions["b"])
    # Input errors above do not produce apparently complete files.
    return settings, clinicians, chunks, labels, predictions, notes


def run_solve(args, session, output, clinicians, chunks, labels, predictions):
    assignments, loads, results = [], [], []
    severity = predictions[""]
    for chunk, patients in chunks.items():
        row = {
            "label": labels[""],
            "chunk_id": chunk,
            "patient_count": len(patients),
            "clinician_count": len(clinicians),
        }
        try:
            optimum = solve_lexicographic(
                session, patients, clinicians, severity.weights, args.clinician_capacity
            )
            row.update(flatten_stages(optimum))
            if optimum.assignment is not None:
                final_loads, _, _ = evaluate(
                    optimum.assignment,
                    patients,
                    clinicians,
                    severity.weights,
                    args.clinician_capacity,
                )
                assignments.extend(
                    {
                        "label": labels[""],
                        "chunk_id": chunk,
                        "patient_id": p,
                        "clinician_id": optimum.assignment[p],
                        "treatment_category": severity.categories[p],
                        "severity_weight": severity.weights[p],
                        "solution_kind": "individual",
                        "certified": optimum.certified,
                        "status": optimum.stages.get("stage2", optimum.stages["stage1"])["status"],
                    }
                    for p in patients
                )
                loads.extend(
                    {"label": labels[""], "chunk_id": chunk, "certified": optimum.certified, **load}
                    for load in final_loads
                )
        except (session.gp.GurobiError, ValueError) as exc:
            row.update(component_optimal=False, error=solver_error(exc, session))
        results.append(row)
        print(
            f"{chunk}: {'certified' if row.get('component_optimal') else 'incomplete'}", flush=True
        )
    write_csv(
        output / "assignments.csv",
        assignments,
        (
            "label",
            "chunk_id",
            "patient_id",
            "clinician_id",
            "treatment_category",
            "severity_weight",
            "solution_kind",
            "certified",
            "status",
        ),
    )
    write_csv(
        output / "clinician_loads.csv",
        loads,
        (
            "label",
            "chunk_id",
            "clinician_id",
            "initial_count",
            "initial_severity",
            "assigned_count",
            "assigned_severity",
            "final_count",
            "final_severity",
            "capacity",
            "certified",
        ),
    )
    write_csv(output / "chunk_results.csv", results)
    complete = all(r.get("component_optimal") for r in results)
    write_json(
        output / "run_status.json",
        {
            "complete": complete,
            "total_chunks": len(chunks),
            "certified_chunks": sum(bool(r.get("component_optimal")) for r in results),
        },
    )
    return 0 if complete else 1


def solver_error(exc, session):
    if isinstance(exc, session.gp.GurobiError):
        return f"Gurobi error code {exc.errno}; inspect license/model configuration locally"
    return str(exc)


def save_pair(
    args, patients, clinicians, labels, predictions, optima, pair, row, assignments, loads
):
    a, b = pair
    changed = sum(a[p] != b[p] for p in patients)
    row.update(
        observed_disagreement_count=changed,
        observed_disagreement_rate=changed / len(patients),
        changed_count=changed,
        changed_proportion=changed / len(patients),
    )
    if args.selection_mode == "matched":
        row["minimum_disagreement_count"] = changed if row["selection_certified"] else None
        row["minimum_disagreement_rate"] = (
            changed / len(patients) if row["selection_certified"] else None
        )
    if args.reference_side:
        ref = args.reference_side.lower()
        ai_assignment = b if ref == "a" else a
        row.update(
            reference_label=labels[ref],
            **reference_regret(
                ai_assignment,
                patients,
                clinicians,
                predictions[ref].weights,
                args.clinician_capacity,
                optima[ref],
            ),
        )
        row["regret_provisional"] = not row["selection_certified"]
    common = {
        key: row[key]
        for key in (
            "pair_id",
            "chunk_id",
            "selection_mode",
            "repeat_id",
            "selection_certified",
            "provisional",
        )
    }
    for p in patients:
        assignments.append(
            {
                **common,
                "patient_id": p,
                "label_A": labels["a"],
                "label_B": labels["b"],
                "category_A": predictions["a"].categories[p],
                "category_B": predictions["b"].categories[p],
                "weight_A": predictions["a"].weights[p],
                "weight_B": predictions["b"].weights[p],
                "selected_clinician_A": a[p],
                "selected_clinician_B": b[p],
                "changed": a[p] != b[p],
                "solution_kind": "matched"
                if args.selection_mode == "matched"
                else "random_optimal",
                "matching_status": row.get("matching_status"),
                "matching_optimal": row.get("matching_optimal"),
                "selection_status_A": row.get("selection_A_status"),
                "selection_status_B": row.get("selection_B_status"),
            }
        )
    for side, assignment in (("a", a), ("b", b)):
        own_loads, _, _ = verify_optimal_set(
            assignment,
            patients,
            clinicians,
            predictions[side].weights,
            args.clinician_capacity,
            optima[side],
        )
        ref_loads = {}
        if args.reference_side:
            ref_loads = {
                r["clinician_id"]: r["final_severity"]
                for r in evaluate(
                    assignment,
                    patients,
                    clinicians,
                    predictions[args.reference_side.lower()].weights,
                    args.clinician_capacity,
                )[0]
            }
        loads.extend(
            {
                **common,
                "side": side.upper(),
                "label": labels[side],
                **load,
                "final_severity_under_reference": ref_loads.get(load["clinician_id"]),
            }
            for load in own_loads
        )


def run_comparison(args, session, output, clinicians, chunks, labels, predictions):
    assignments, loads, results, streams = [], [], [], []
    component_assignments, component_loads, component_results = [], [], []
    selection_incumbents = []
    for chunk, patients in chunks.items():
        base = {
            "pair_id": labels["a"] + "_vs_" + labels["b"],
            "chunk_id": chunk,
            "patient_count": len(patients),
            "clinician_count": len(clinicians),
            "selection_mode": args.selection_mode,
        }
        optima = {}
        try:
            for side in ("a", "b"):
                optima[side] = solve_lexicographic(
                    session,
                    patients,
                    clinicians,
                    predictions[side].weights,
                    args.clinician_capacity,
                )
                base.update(flatten_stages(optima[side], side.upper() + "_"))
                component_results.append(
                    {
                        "label": labels[side],
                        "side": side.upper(),
                        "chunk_id": chunk,
                        "patient_count": len(patients),
                        **flatten_stages(optima[side]),
                    }
                )
                if optima[side].assignment is not None:
                    component_assignments.extend(
                        {
                            "label": labels[side],
                            "side": side.upper(),
                            "chunk_id": chunk,
                            "patient_id": p,
                            "clinician_id": optima[side].assignment[p],
                            "treatment_category": predictions[side].categories[p],
                            "severity_weight": predictions[side].weights[p],
                            "solution_kind": "individual",
                            "certified": optima[side].certified,
                        }
                        for p in patients
                    )
                    component_loads.extend(
                        {
                            "label": labels[side],
                            "side": side.upper(),
                            "chunk_id": chunk,
                            "certified": optima[side].certified,
                            **load,
                        }
                        for load in evaluate(
                            optima[side].assignment,
                            patients,
                            clinicians,
                            predictions[side].weights,
                            args.clinician_capacity,
                        )[0]
                    )
            if (
                all(o.certified for o in optima.values())
                and optima["a"].count_range != optima["b"].count_range
            ):
                raise ValueError("Count-stage optimum differs under identical operational inputs")
        except (session.gp.GurobiError, ValueError) as exc:
            base["error"] = solver_error(exc, session)
        for repeat in range(1, args.selection_repeats + 1):
            row = {
                **base,
                "repeat_id": repeat,
                "selection_certified": False,
                "provisional": False,
                "observed_disagreement_count": None,
                "observed_disagreement_rate": None,
                "minimum_disagreement_count": None,
                "minimum_disagreement_rate": None,
                "reference_label": None,
                "G_reference": None,
                "AI_reference_gap": None,
                "raw_regret": None,
                "reported_regret": None,
                **flatten_status(None, "matching_"),
                **flatten_status(None, "selection_A_"),
                **flatten_status(None, "selection_B_"),
            }
            pending_assignments, pending_loads = [], []
            if (
                "error" not in base
                and len(optima) == 2
                and all(o.certified for o in optima.values())
            ):
                try:
                    if args.selection_mode == "matched":
                        a, b, status = match_optima(
                            session,
                            patients,
                            clinicians,
                            predictions["a"].weights,
                            predictions["b"].weights,
                            args.clinician_capacity,
                            optima["a"],
                            optima["b"],
                        )
                        row.update(
                            flatten_status(status, "matching_"),
                            selection_certified=status["optimal"],
                        )
                    else:
                        pair, statuses = {}, {}
                        for side in ("a", "b"):
                            values, metadata = coefficients(
                                args.selection_seed,
                                chunk,
                                repeat,
                                labels[side],
                                patients,
                                [c.name for c in clinicians],
                            )
                            streams.append(
                                {
                                    "chunk_id": chunk,
                                    "repeat_id": repeat,
                                    "label": labels[side],
                                    **metadata,
                                }
                            )
                            pair[side], statuses[side] = random_optimum(
                                session,
                                patients,
                                clinicians,
                                predictions[side].weights,
                                args.clinician_capacity,
                                optima[side],
                                values,
                            )
                            if pair[side] is not None:
                                selection_incumbents.extend(
                                    {
                                        "label": labels[side],
                                        "side": side.upper(),
                                        "chunk_id": chunk,
                                        "repeat_id": repeat,
                                        "selection_mode": "random",
                                        "patient_id": p,
                                        "clinician_id": pair[side][p],
                                        "solution_kind": "random_optimal",
                                        "component_optimal": True,
                                        "selection_optimal": statuses[side]["optimal"],
                                        "selection_status": statuses[side]["status"],
                                    }
                                    for p in patients
                                )
                            row.update(
                                flatten_status(statuses[side], "selection_" + side.upper() + "_")
                            )
                        a, b = pair["a"], pair["b"]
                        row["selection_certified"] = all(s["optimal"] for s in statuses.values())
                    if a is not None and b is not None:
                        row["provisional"] = not row["selection_certified"]
                        save_pair(
                            args,
                            patients,
                            clinicians,
                            labels,
                            predictions,
                            optima,
                            (a, b),
                            row,
                            pending_assignments,
                            pending_loads,
                        )
                        assignments.extend(pending_assignments)
                        loads.extend(pending_loads)
                except (session.gp.GurobiError, ValueError) as exc:
                    row.update(selection_certified=False, error=solver_error(exc, session))
                    # An invalid candidate contributes no claimed metrics/allocations.
                    for key in (
                        "observed_disagreement_count",
                        "observed_disagreement_rate",
                        "minimum_disagreement_count",
                        "minimum_disagreement_rate",
                        "changed_count",
                        "changed_proportion",
                        "G_reference",
                        "AI_reference_gap",
                        "raw_regret",
                        "reported_regret",
                    ):
                        row[key] = None
            results.append(row)
        print(
            f"{chunk}: {sum(bool(r['selection_certified']) for r in results if r['chunk_id'] == chunk)}/{args.selection_repeats} certified selections",
            flush=True,
        )
    write_csv(
        output / "comparison_assignments.csv",
        assignments,
        (
            "pair_id",
            "chunk_id",
            "selection_mode",
            "repeat_id",
            "patient_id",
            "label_A",
            "label_B",
            "category_A",
            "category_B",
            "weight_A",
            "weight_B",
            "selected_clinician_A",
            "selected_clinician_B",
            "changed",
            "selection_certified",
            "solution_kind",
            "provisional",
            "matching_status",
            "matching_optimal",
            "selection_status_A",
            "selection_status_B",
        ),
    )
    write_csv(
        output / "comparison_clinician_loads.csv",
        loads,
        (
            "pair_id",
            "chunk_id",
            "selection_mode",
            "repeat_id",
            "side",
            "clinician_id",
            "initial_count",
            "initial_severity",
            "assigned_count",
            "assigned_severity",
            "final_count",
            "final_severity",
            "capacity",
        ),
    )
    write_csv(output / "comparison_results.csv", results)
    write_csv(
        output / "component_assignments.csv",
        component_assignments,
        ("label", "side", "chunk_id", "patient_id", "clinician_id", "solution_kind", "certified"),
    )
    write_csv(
        output / "component_clinician_loads.csv",
        component_loads,
        (
            "label",
            "side",
            "chunk_id",
            "clinician_id",
            "initial_count",
            "initial_severity",
            "assigned_count",
            "assigned_severity",
            "final_count",
            "final_severity",
            "capacity",
            "certified",
        ),
    )
    write_csv(
        output / "component_results.csv",
        component_results,
        ("label", "side", "chunk_id", "component_optimal"),
    )
    write_json(output / "selection_streams.json", streams)
    write_csv(
        output / "selection_incumbents.csv",
        selection_incumbents,
        (
            "label",
            "side",
            "chunk_id",
            "repeat_id",
            "selection_mode",
            "patient_id",
            "clinician_id",
            "solution_kind",
            "component_optimal",
            "selection_optimal",
            "selection_status",
        ),
    )
    summary = comparison_summary(results, chunks, args.selection_mode, args.selection_repeats)
    write_json(output / "comparison_summary.json", summary)
    return 0 if summary["complete"] else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    try:
        if args.command == "chunk":
            return create_chunks(args)
        settings, clinicians, chunks, labels, predictions, notes = preflight(args)
        output = fresh_directory(args.output)
        with SolverSession(settings) as session:
            config = base_config(
                args,
                clinicians,
                settings,
                chunks,
                {side: p.metadata for side, p in predictions.items()},
                session,
            )
            config["labels"] = labels
            if args.command == "solve":
                write_json(output / "run_config.json", config)
                return run_solve(args, session, output, clinicians, chunks, labels, predictions)
            config.update(
                note_hash_consistency=notes,
                reference_side=args.reference_side,
                selection_mode=args.selection_mode,
                selection_seed=args.selection_seed,
                selection_repeats=args.selection_repeats,
                selection_seed_active=args.selection_mode == "random",
                selection_repeats_active=args.selection_mode == "random",
                tie_policy="solver_selected_closest_pair"
                if args.selection_mode == "matched"
                else "independent_random_linear_objective_on_certified_optimal_sets",
                matching_objective="minimize patient disagreement"
                if args.selection_mode == "matched"
                else None,
                random_algorithm=RANDOM_ALGORITHM if args.selection_mode == "random" else None,
                random_sampling_warning="Not uniform sampling over all optimal allocations"
                if args.selection_mode == "random"
                else None,
                reference_interpretation="Matching uses reference information when a reference side is supplied: retrospective, matching-conditioned regret, not a deployment policy; minimum disagreement does not imply minimum regret"
                if args.selection_mode == "matched"
                else "AI selection is independent of reference; retrospective evaluation of the saved randomized optimum",
                certification_policy="Only certified complete selections enter completed aggregates; feasible interrupted candidates are provisional",
            )
            write_json(output / "comparison_config.json", config)
            return run_comparison(args, session, output, clinicians, chunks, labels, predictions)
    except (FileNotFoundError, ValueError, SolverDependencyError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if output is not None:
            write_json(
                output / "failure_status.json",
                {"complete": False, "exit_code": 2, "error": str(exc)},
            )
        return 2
