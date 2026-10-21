"""Intersect valid model outputs before shuffling; no inference or solver calls."""

import random

from .cohort import normalize_id
from .experiment_inputs import integer, keys, load_score, number, read_yaml, schema, unique
from .inputs import file_hash, read_table, unique_ids
from .proxy_inputs import membership, select
from .proxy_uncertainty import configuration, validate_probability_row
from .reporting import fresh_directory, write_csv, write_json


def run(args):
    size = integer(args.chunk_size, "chunk-size", 1)
    seed = integer(args.seed, "seed")
    if args.uncertainty_config and not args.require_probabilities:
        raise ValueError("--uncertainty-config requires --require-probabilities")
    raw = read_yaml(args.config)
    schema(
        raw,
        ["category_order", "category_weights", "category_aliases", "inputs"],
        ["category_order", "category_weights", "inputs"],
    )
    order = raw["category_order"]
    unique(order, "category_order")
    weights = {k: number(v, "category weight") for k, v in raw["category_weights"].items()}
    if set(weights) != set(order):
        raise ValueError("category_weights must match category_order")
    aliases = raw.get("category_aliases", {})
    if set(aliases.values()) - set(order) or any(k in order and k != v for k, v in aliases.items()):
        raise ValueError("Invalid category aliases")
    scoring = {
        "category_weights": weights,
        "category_aliases": aliases,
        "bounds": [min(weights.values()), max(weights.values())],
        "scale": "proxy complexity",
    }
    unique([normalize_id(e["id"]) for e in raw["inputs"]], "input IDs")
    unique([*args.models, args.benchmark], "candidate/benchmark IDs")
    entries = select(
        {normalize_id(e["id"]): e for e in raw["inputs"]},
        [*args.models, args.benchmark],
        "model IDs",
    )
    settings = (
        configuration(args.uncertainty_config, ["stochastic"])[0]["probabilities"]
        if args.require_probabilities
        else None
    )
    loaded, valid, reasons, summaries = {}, {}, {}, []
    hashes = {str(args.config.resolve()): file_hash(args.config)}
    if args.uncertainty_config:
        hashes[str(args.uncertainty_config.resolve())] = file_hash(args.uncertainty_config)
    for mid, entry in entries.items():
        keys(
            entry,
            [
                "id",
                "path",
                "sheet",
                "id_column",
                "category_column",
                "probability_path",
                "probability_id_column",
                "probability_columns",
                "probability_sheet",
            ],
            ["id", "path", "id_column", "category_column"],
        )
        adapted = {
            k: v
            for k, v in entry.items()
            if k != "category_column" and not k.startswith("probability_")
        }
        adapted.update(value_column=entry["category_column"], value_type="category")
        data = load_score(adapted, args.config.resolve().parent, scoring)
        loaded[mid] = data
        hashes[data["entry"]["path"]] = data["sha256"]
        valid[mid] = set(data["weights"])
        reasons[mid] = {p: "failed_or_invalid_categorical_prediction" for p in data["invalid"]}
        corrections = disagreements = 0
        if args.require_probabilities and mid != args.benchmark:
            path = (
                args.config.resolve().parent / entry.get("probability_path", entry["path"])
            ).resolve()
            columns = entry.get("probability_columns", settings["columns"])
            if not isinstance(columns, dict) or len(columns) != len(order):
                raise ValueError(f"{mid}: probability_columns must map every category")
            columns = {aliases.get(k, k): v for k, v in columns.items()}
            if set(columns) != set(order) or len(set(columns.values())) != len(order):
                raise ValueError(f"{mid}: ambiguous probability_columns")
            id_column = entry.get("probability_id_column", entry["id_column"])
            frame = unique_ids(
                read_table(path, entry.get("probability_sheet", entry.get("sheet"))), id_column
            )
            if set(columns.values()) - set(frame.columns):
                raise ValueError(
                    f"{mid}: missing probability columns; configure probability_path/columns with full distributions"
                )
            rows = {r[id_column]: r for r in frame.to_dict("records")}
            hashes[str(path)] = file_hash(path)
            for pid in sorted(valid[mid]):
                if pid not in rows:
                    reasons[mid][pid] = "missing_probabilities"
                    continue
                row = rows[pid]
                try:
                    for name in ("success", "successful", "status", "classification_status"):
                        if name in row:
                            allowed = (
                                {"true", "1", "1.0"}
                                if name in ("success", "successful")
                                else {"success", "successful", "completed", "validated", "ok"}
                            )
                            if str(row[name]).strip().lower() not in allowed:
                                raise ValueError("Failed probability inference")
                    _, corrected, disagreement = validate_probability_row(
                        row, columns, order, data["categories"][pid], settings
                    )
                    corrections += corrected
                    disagreements += disagreement
                except ValueError:
                    reasons[mid][pid] = "failed_or_invalid_probabilities"
            valid[mid] -= set(reasons[mid])
        summaries.append(
            {
                "model_id": mid,
                "successful_categories": len(data["weights"]),
                "eligible_predictions": len(valid[mid]),
                "invalid_category_rows": len(data["invalid"]),
                "probabilities_required": args.require_probabilities and mid != args.benchmark,
                "renormalizable_rows": corrections,
                "point_argmax_disagreements": disagreements,
            }
        )
    universe = set().union(*(set(d["weights"]) | set(d["invalid"]) for d in loaded.values()))
    eligibility = None
    if args.eligible_chunks:
        previous, previous_hashes = membership(args.eligible_chunks)
        eligibility = {p for patients in previous.values() for p in patients}
        universe |= eligibility
        hashes.update(previous_hashes)
    common = set.intersection(*valid.values())
    if eligibility is not None:
        common &= eligibility
    if not common:
        raise ValueError("No common valid clients remain across all candidates and benchmark")
    # Match the downstream source consistency requirement before creating membership.
    for pid in common:
        seen = {}
        for data in loaded.values():
            for name, value in data["notes"][pid].items():
                if name in seen and seen[name] != value:
                    raise ValueError(f"Source note hash mismatch for de-identified client {pid}")
                seen[name] = value
    patients = sorted(common)
    random.Random(seed).shuffle(patients)
    remainder = len(patients) % size
    dropped = patients[-remainder:] if args.drop_remainder and remainder else []
    selected = patients[:-remainder] if dropped else patients
    if not selected:
        raise ValueError("Fewer common clients than chunk-size; omit --drop-remainder to keep them")
    rows, chunk_summary = [], []
    for start in range(0, len(selected), size):
        cid = f"chunk_{start // size + 1:04d}"
        group = selected[start : start + size]
        rows.extend(
            {"patient_id": p, "chunk_id": cid, "within_chunk_order": j} for j, p in enumerate(group)
        )
        chunk_summary.append(
            {"chunk_id": cid, "patient_count": len(group), "is_remainder": len(group) < size}
        )
    exclusions = []
    for pid in sorted(universe - common):
        if eligibility is not None and pid not in eligibility:
            exclusions.append(
                {"patient_id": pid, "model_id": "", "reason": "outside_eligible_population"}
            )
        else:
            for mid in entries:
                if pid not in valid[mid]:
                    exclusions.append(
                        {
                            "patient_id": pid,
                            "model_id": mid,
                            "reason": reasons[mid].get(pid, "missing_prediction"),
                        }
                    )
    exclusions.extend(
        {"patient_id": p, "model_id": "", "reason": "dropped_remainder"} for p in dropped
    )
    output = fresh_directory(args.output)
    write_csv(output / "chunks.csv", rows)
    write_csv(output / "chunk_summary.csv", chunk_summary)
    write_csv(output / "common_successful_clients.csv", [{"patient_id": p} for p in sorted(common)])
    write_csv(output / "exclusions.csv", exclusions, ["patient_id", "model_id", "reason"])
    write_csv(output / "model_coverage.csv", summaries)
    write_json(
        output / "cohort_report.json",
        {
            "models": args.models,
            "benchmark": args.benchmark,
            "probability_settings": settings,
            "input_hashes": hashes,
            "seed": seed,
            "chunk_size": size,
            "population_filter": "IDs from eligible-chunks"
            if eligibility is not None
            else "all prediction IDs",
            "common_valid_clients": len(common),
            "included_clients": len(selected),
            "excluded_clients": len(universe - common),
            "dropped_remainder": len(dropped),
            "chunk_count": len(chunk_summary),
            "remainder_policy": "drop" if args.drop_remainder else "keep",
            "membership_sha256": file_hash(output / "chunks.csv"),
        },
    )
    print(
        f"{len(common)} common valid clients; {len(selected)} included in {len(chunk_summary)} chunks. Use --chunks {output / 'chunks.csv'}"
    )
    return 0
