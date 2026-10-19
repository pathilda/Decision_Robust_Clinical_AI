"""Strict categorical registry and unchanged chunk membership for proxy analysis."""

import json
from pathlib import Path

from .cohort import normalize_id
from .experiment_inputs import (
    digest,
    integer,
    keys,
    load_score,
    number,
    read_yaml,
    schema,
    unique,
)
from .inputs import file_hash, load_membership

PROFILES = ("balanced", "unbalanced", "highly_unbalanced")


def select(values, requested, name):
    requested = list(values) if requested is None else requested
    unique(requested, name)
    if set(requested) - set(values):
        raise ValueError(f"Unknown {name}: {sorted(set(requested) - set(values))}")
    return {key: values[key] for key in requested}


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def membership(path):
    path = Path(path).resolve()
    hashes = {str(path): file_hash(path)}
    if path.suffix.lower() != ".json":
        return load_membership(path), hashes
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=no_duplicate_keys)
    if "membership" in value:
        keys(value, ["schema_version", "membership", "sha256"], ["membership"])
        source = (path.parent / value["membership"]).resolve()
        if value.get("sha256") and value["sha256"] != file_hash(source):
            raise ValueError("Chunk membership hash mismatch")
        chunks = load_membership(source)
        hashes[str(source)] = file_hash(source)
    else:
        keys(value, ["schema_version", "chunks"], ["chunks"])
        if not isinstance(value["chunks"], dict):
            raise ValueError("JSON chunks must map chunk IDs to patient ID lists")
        chunks = {}
        for cid, patients in value["chunks"].items():
            cid = normalize_id(cid)
            if cid in chunks or not isinstance(patients, list):
                raise ValueError("Duplicate chunk ID or invalid patient list")
            chunks[cid] = sorted(normalize_id(p) for p in patients)
            unique(chunks[cid], "patients within chunk")
    unique(list(chunks), "chunks")
    unique([p for patients in chunks.values() for p in patients], "patients across chunks")
    return dict(sorted(chunks.items())), hashes


def profiles(path, weights):
    raw = read_yaml(path)
    keys(raw, ["schema_version", "profiles", "metadata"], ["profiles"])
    entries = raw["profiles"]
    if isinstance(entries, dict):
        entries = [{"id": key, **value} for key, value in entries.items()]
    result = {}
    for entry in entries:
        keys(entry, ["id", "clinicians", "metadata"], ["id", "clinicians"])
        pid = normalize_id(entry["id"])
        if pid in result:
            raise ValueError("Duplicate profile ID")
        clinicians = []
        for c in entry["clinicians"]:
            keys(
                c,
                ["id", "initial_count", "initial_complexity", "capacity"],
                ["id", "initial_count", "initial_complexity", "capacity"],
            )
            count = integer(c["initial_count"], "initial_count")
            cap = integer(c["capacity"], "capacity")
            load = number(c["initial_complexity"], "initial_complexity")
            if (
                cap < count
                or not count * min(weights) - 1e-9 <= load <= count * max(weights) + 1e-9
            ):
                raise ValueError(
                    f"Profile {pid}: capacity/count or scale-incompatible initial load"
                )
            clinicians.append(
                {
                    "id": normalize_id(c["id"]),
                    "initial_count": count,
                    "initial_complexity": load,
                    "capacity": cap,
                }
            )
        unique([c["id"] for c in clinicians], "clinician IDs")
        result[pid] = sorted(clinicians, key=lambda c: c["id"])
    if set(result) != set(PROFILES):
        raise ValueError(f"Profile file must contain exactly {PROFILES}")
    identities = [tuple(c["id"] for c in result[p]) for p in PROFILES]
    if len(set(identities)) != 1:
        raise ValueError("Clinician identities must be identical across profiles")
    return {p: result[p] for p in PROFILES}


def agreement(candidate, benchmark, order):
    """Full ordinal order, independent of numeric complexity weights."""
    n = len(candidate)
    if not n or n != len(benchmark):
        raise ValueError("Agreement requires the same nonempty cohort")
    accuracy = 100 * sum(a == b for a, b in zip(candidate, benchmark)) / n
    if len(order) < 2:
        return accuracy, None, "undefined: fewer than two ordered categories"
    positions = {c: i for i, c in enumerate(order)}
    a, b = [positions[c] for c in candidate], [positions[c] for c in benchmark]
    observed = sum((i - j) ** 2 for i, j in zip(a, b)) / n
    expected = (
        sum(
            a.count(i) * b.count(j) * (i - j) ** 2
            for i in range(len(order))
            for j in range(len(order))
        )
        / n**2
    )
    if expected == 0:
        return accuracy, None, "undefined: zero expected weighted disagreement"
    return accuracy, 1 - observed / expected, "defined"


def prepare(args):
    raw = read_yaml(args.config)
    schema(
        raw,
        ["category_order", "category_weights", "category_aliases", "inputs"],
        ["category_order", "category_weights", "inputs"],
    )
    order = raw["category_order"]
    unique(order, "ordered categories")
    weights = {k: number(v, "category weight") for k, v in raw["category_weights"].items()}
    if set(weights) != set(order):
        raise ValueError("category_weights must match the full category_order")
    aliases = raw.get("category_aliases", {})
    if set(aliases.values()) - set(order) or any(k in order and k != v for k, v in aliases.items()):
        raise ValueError("category_aliases must map unambiguously to category_order")
    scoring = {
        "category_weights": weights,
        "category_aliases": aliases,
        "bounds": [min(weights.values()), max(weights.values())],
        "scale": "proxy complexity",
    }
    entries = raw["inputs"]
    unique([normalize_id(e["id"]) for e in entries], "input IDs")
    entries = {normalize_id(e["id"]): e for e in entries}
    unique(args.models, "models")
    if args.benchmark in args.models:
        raise ValueError("Proxy benchmark must not be included in --models")
    chosen = select(entries, [*args.models, args.benchmark], "configuration IDs")
    chunks, hashes = membership(args.chunks)
    chunks = select(chunks, args.chunk_ids, "chunk IDs")
    patients = sorted(p for ps in chunks.values() for p in ps)
    loaded = {}
    for mid, entry in chosen.items():
        keys(
            entry,
            ["id", "path", "sheet", "id_column", "category_column"],
            ["id", "path", "id_column", "category_column"],
        )
        adapted = {k: v for k, v in entry.items() if k != "category_column"}
        adapted.update(value_column=entry["category_column"], value_type="category")
        data = load_score(adapted, args.config.resolve().parent, scoring)
        missing = set(patients) - set(data["weights"])
        if missing:
            reasons = {p: data["invalid"].get(p, "missing prediction") for p in sorted(missing)}
            raise ValueError(f"{mid}: missing/failed categorical assessments: {reasons}")
        loaded[mid] = data
        hashes[data["entry"]["path"]] = data["sha256"]
    # Common source-note hashes, when available, must agree across configurations.
    for p in patients:
        seen = {}
        for mid, data in loaded.items():
            for key, val in data["notes"][p].items():
                if key in seen and seen[key] != val:
                    raise ValueError(f"Source note hash mismatch for patient {p} ({mid})")
                seen[key] = val
    selected_profiles = select(
        profiles(args.profiles, list(weights.values())),
        args.starting_parameters,
        "starting parameters",
    )
    if args.selection_seeds is not None:
        if args.n_samples is not None or args.sample_seed is not None:
            raise ValueError(
                "--selection-seeds is mutually exclusive with --n-samples/--sample-seed"
            )
        seeds = args.selection_seeds
        unique(seeds, "selection seeds")
    else:
        n = integer(30 if args.n_samples is None else args.n_samples, "n-samples", 1)
        seed = 42 if args.sample_seed is None else args.sample_seed
        seeds = list(range(seed, seed + n))
    for path in (args.config, args.profiles):
        hashes[str(path.resolve())] = file_hash(path)
    instances = [
        {
            "chunk_id": cid,
            "profile_id": pid,
            "patients": ps,
            "clinicians": cs,
            "feasible": len(ps) <= sum(c["capacity"] - c["initial_count"] for c in cs),
        }
        for cid, ps in chunks.items()
        for pid, cs in selected_profiles.items()
    ]
    accuracy = []
    for mid in args.models:
        acc, qwk, status = agreement(
            [loaded[mid]["categories"][p] for p in patients],
            [loaded[args.benchmark]["categories"][p] for p in patients],
            order,
        )
        accuracy.append(
            {
                "model_id": mid,
                "benchmark_id": args.benchmark,
                "benchmark_kind": "proxy",
                "n_patients": len(patients),
                "exact_accuracy_pct": acc,
                "qwk": qwk,
                "qwk_status": status,
            }
        )
    resolved = {
        "schema_version": 1,
        "workflow": "proxy-analysis",
        "models": args.models,
        "benchmark_id": args.benchmark,
        "benchmark_kind": "proxy",
        "chunks": chunks,
        "profiles": selected_profiles,
        "category_order": order,
        "category_weights": weights,
        "category_aliases": aliases,
        "selection_seeds": seeds,
        "benchmark_seed": args.benchmark_seed,
        "chunk_seed": args.chunk_seed,
        "objective": args.objective,
        "input_hashes": hashes,
        "inputs": {mid: data["entry"] for mid, data in loaded.items()},
        "optimality_tolerance": args.optimality_tolerance,
    }
    return {
        "resolved": resolved,
        "loaded": loaded,
        "instances": instances,
        "accuracy": accuracy,
        "cohort_hash": digest(patients),
    }
