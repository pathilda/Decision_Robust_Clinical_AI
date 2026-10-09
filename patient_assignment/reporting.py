"""Mode-neutral tables, complete provenance, and certified-only summaries."""

from __future__ import annotations

import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .inputs import WEIGHTS, file_hash
from .optimization import VERIFICATION_TOLERANCE


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be an empty or absent directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows, required_columns=()):
    columns = list(dict.fromkeys([*required_columns, *(key for row in rows for key in row)]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: str(v).lower() if isinstance(v, bool) else v for k, v in row.items()}
            )


def base_config(args, clinicians, settings, chunks, predictions, session):
    try:
        package_version = version("clinical-treatment-classification")
    except PackageNotFoundError:
        package_version = "uninstalled checkout"
    warnings = []
    for c in clinicians:
        if c.initial_severity < c.initial_count or c.initial_severity > 3 * c.initial_count:
            warnings.append(
                f"{c.name}: initial severity is outside [initial_count, 3*initial_count]; supplied units/state are retained, no initial patient records available"
            )
    return {
        "schema_version": 1,
        "command": args.command,
        "options": vars(args),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "membership": {"path": str(args.chunks.resolve()), "sha256": file_hash(args.chunks)},
        "ordered_chunks": {key: ids for key, ids in chunks.items()},
        "ordered_clinicians": [
            {
                "name": c.name,
                "initial_count": c.initial_count,
                "initial_severity": str(c.initial_severity),
                "initial_severity_input": c.initial_severity_input,
            }
            for c in clinicians
        ],
        "fixed_weights": WEIGHTS,
        "predictions": predictions,
        "solver_parameters": settings.parameters(),
        "verification_tolerance": VERIFICATION_TOLERANCE,
        "python_version": platform.python_version(),
        "gurobi_version": list(session.gp.gurobi.version()),
        "package_version": package_version,
        "allocation_package_version": __version__,
        "independent_chunks": "Each chunk starts from the same supplied initial state independently",
        "initial_state_assumption": "Initial patients are outside new membership; aggregates cannot verify overlap",
        "warnings": warnings,
    }


def flatten_stages(optimum, prefix=""):
    row = {
        prefix + "component_optimal": optimum.certified,
        prefix + "count_range": optimum.count_range,
        prefix + "severity_range": optimum.severity_range,
    }
    for stage in ("stage1", "stage2"):
        for key in (
            "status",
            "status_code",
            "optimal",
            "has_incumbent",
            "objective",
            "bound",
            "gap",
            "runtime",
        ):
            row[f"{prefix}{stage}_{key}"] = optimum.stages.get(stage, {}).get(key)
    return row


def flatten_status(status, prefix):
    return {
        prefix + key: status.get(key) if status else None
        for key in (
            "status",
            "status_code",
            "optimal",
            "has_incumbent",
            "objective",
            "bound",
            "gap",
            "runtime",
        )
    }


def describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def comparison_summary(rows, chunks, mode, repeats):
    certified = [r for r in rows if r.get("selection_certified")]
    per_chunk = []
    for chunk in chunks:
        selected = [r for r in certified if r["chunk_id"] == chunk]
        per_chunk.append(
            {
                "chunk_id": chunk,
                "completed_selections": len(selected),
                "requested_selections": repeats,
                "disagreement_count": describe(
                    [r["observed_disagreement_count"] for r in selected]
                ),
                "disagreement_rate": describe([r["observed_disagreement_rate"] for r in selected]),
                "regret": describe([r.get("reported_regret") for r in selected]),
            }
        )
    complete_chunks = [r for r in per_chunk if r["completed_selections"] == repeats]
    provisional_chunks = {
        chunk
        for chunk in chunks
        if any(row["chunk_id"] == chunk and row.get("provisional") for row in rows)
    }
    successful_ids = {r["chunk_id"] for r in complete_chunks}
    denominator = sum(r["patient_count"] for r in certified)
    return {
        "selection_mode": mode,
        "complete": len(certified) == len(chunks) * repeats,
        "total_chunks": len(chunks),
        "successful_chunks": len(complete_chunks),
        "failed_chunks": len(set(chunks) - successful_ids - provisional_chunks),
        "provisional_chunks": len(provisional_chunks - successful_ids),
        "incomplete_chunks": len(chunks) - len(complete_chunks),
        "requested_selections": len(chunks) * repeats,
        "certified_selections": len(certified),
        "coverage": {
            "membership_patients": sum(map(len, chunks.values())),
            "patients_in_complete_chunks": sum(len(chunks[r["chunk_id"]]) for r in complete_chunks),
        },
        "within_chunk": per_chunk,
        "across_chunks_weighting": "Equal weight per chunk mean; only complete chunks. Repeats are conditional sensitivity, not independent clinical samples.",
        "mean_per_chunk_disagreement_rate": describe(
            [r["disagreement_rate"]["mean"] for r in complete_chunks]
        )["mean"],
        "across_chunk_mean_disagreement_rate": describe(
            [r["disagreement_rate"]["mean"] for r in complete_chunks]
        ),
        "across_chunk_mean_regret": describe([r["regret"]["mean"] for r in complete_chunks]),
        "regret_over_certified_selections": describe([r.get("reported_regret") for r in certified]),
        "pooled_patient_disagreement": sum(r["observed_disagreement_count"] for r in certified)
        / denominator
        if denominator
        else None,
        "pooled_weighting": "sum(changed) / sum(N) over certified chunk/repeat rows; repeated patients are not additional clinical samples",
        "excluded_uncertified_rows": len(rows) - len(certified),
    }
