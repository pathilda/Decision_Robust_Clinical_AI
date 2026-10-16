"""Validated registry, operational profiles and uncertainty, independent of Gurobi."""

# Configuration validation uses ValueError consistently at the CLI boundary.
# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import yaml

from .cohort import normalize_id
from .experiment_cli import DEFAULTS
from .inputs import WEIGHTS, file_hash, load_membership, parse_clinicians, read_table, unique_ids


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seed_for(master, *parts):
    return int(digest([master, *parts])[:16], 16)


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    try:
        value = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return value


def keys(value, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError("Expected a mapping")
    if set(value) - set(allowed):
        raise ValueError(f"Unknown keys: {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        raise ValueError(f"Missing keys: {sorted(set(required) - set(value))}")


def schema(value, allowed, required=()):
    keys(value, ["schema_version", *allowed], ["schema_version", *required])
    if value["schema_version"] != 1:
        raise ValueError("Only schema_version: 1 is supported")


def number(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def integer(value, name, minimum=0):
    result = number(value, name, minimum)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result)


def path_at(value, base):
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def unique(items, name):
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be nonempty with unique IDs")


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def generate_profiles(args):
    clinicians = sorted(
        parse_clinicians(args.clinician, args.clinician_capacity), key=lambda c: c.name
    )
    lo, hi = number(args.score_min, "score-min"), number(args.score_max, "score-max")
    if hi < lo or len({c.initial_count for c in clinicians}) != 1:
        raise ValueError("Generator requires ordered finite score bounds and equal initial counts")
    factors = [number(a, "imbalance factor") for a in args.imbalance_factors]
    unique(factors, "imbalance factors")
    if max(factors) > 1:
        raise ValueError("Imbalance factors must be in [0,1]")
    seed = integer(args.profile_seed, "profile seed")
    count = clinicians[0].initial_count
    for c in clinicians:
        if not count * lo <= float(c.initial_severity) <= count * hi:
            raise ValueError("Initial loads must be compatible with score bounds")
    total = math.fsum(float(c.initial_severity) for c in clinicians)
    mean = total / len(clinicians)
    if not count * lo <= mean <= count * hi:
        raise ValueError("Impossible equal-load baseline")
    directions = (
        [0.0]
        if len(clinicians) == 1
        else [-1 + 2 * j / (len(clinicians) - 1) for j in range(len(clinicians))]
    )
    random.Random(seed).shuffle(directions)
    bounds = [
        (count * hi - mean) / d if d > 0 else (count * lo - mean) / d for d in directions if d
    ]
    amplitude = min(bounds) if bounds else 0.0
    profiles = []
    for a in factors:
        values = [mean + a * amplitude * d for d in directions]
        profiles.append(
            {
                "id": {0: "balanced", 0.5: "unbalanced", 1: "highly_unbalanced"}.get(
                    a, f"factor_{a:g}"
                ),
                "clinicians": [
                    {
                        "id": c.name,
                        "initial_count": c.initial_count,
                        "initial_complexity": value,
                        "capacity": args.clinician_capacity,
                    }
                    for c, value in zip(clinicians, values)
                ],
                "metadata": {
                    "synthetic": True,
                    "factor": a,
                    "seed": seed,
                    "initial_load_range": max(values) - min(values),
                },
            }
        )
    return {
        "schema_version": 1,
        "profiles": profiles,
        "metadata": {
            "score_min": lo,
            "score_max": hi,
            "total_initial_complexity": total,
            "amplitude_max": amplitude,
            "identical_profiles": amplitude == 0,
            "directions": dict(zip([c.name for c in clinicians], directions)),
            "integer_projection": False,
        },
    }


def load_profiles(path, bounds):
    value = read_yaml(path)
    schema(value, ["profiles", "metadata"], ["profiles"])
    result = {}
    for profile in value["profiles"]:
        keys(profile, ["id", "clinicians", "metadata"], ["id", "clinicians"])
        pid = normalize_id(profile["id"])
        if pid in result:
            raise ValueError(f"Duplicate profile ID: {pid}")
        clinicians = []
        for c in profile["clinicians"]:
            keys(
                c,
                ["id", "initial_count", "initial_complexity", "capacity"],
                ["id", "initial_count", "initial_complexity", "capacity"],
            )
            n = integer(c["initial_count"], "initial_count")
            cap = integer(c["capacity"], "capacity")
            load = number(c["initial_complexity"], "initial_complexity")
            if n > cap or (n == 0 and load != 0):
                raise ValueError(f"Invalid count/load/capacity in profile {pid}")
            if bounds and not n * bounds[0] - 1e-9 <= load <= n * bounds[1] + 1e-9:
                raise ValueError(f"Initial load incompatible with score scale in profile {pid}")
            clinicians.append(
                {
                    "id": normalize_id(c["id"]),
                    "initial_count": n,
                    "initial_complexity": load,
                    "capacity": cap,
                }
            )
        unique([c["id"] for c in clinicians], "clinicians")
        result[pid] = {
            **profile,
            "id": pid,
            "clinicians": sorted(clinicians, key=lambda c: c["id"]),
        }
    unique(list(result), "profiles")
    return result


INPUT_KEYS = [
    "id",
    "path",
    "sheet",
    "id_column",
    "value_column",
    "value_type",
    "scale",
    "normalization",
    "provenance",
    "request_id_column",
    "request_id",
]


def load_score(entry, base, scoring):
    keys(entry, INPUT_KEYS, ["id", "path", "id_column", "value_column", "value_type"])
    entry = dict(entry)
    entry["id"] = normalize_id(entry["id"])
    entry["path"] = path_at(entry["path"], base)
    if entry["value_type"] not in {"category", "numeric"}:
        raise ValueError("value_type must be category or numeric")
    if entry.get("scale", scoring["scale"]) != scoring["scale"]:
        raise ValueError(
            f"Incompatible scale for {entry['id']}; explicitly convert to the common scale"
        )
    frame = unique_ids(read_table(entry["path"], entry.get("sheet")), entry["id_column"])
    column = entry["value_column"]
    if column not in frame:
        raise ValueError(f"Missing score column {column!r}")
    norm = entry.get("normalization", {"multiplier": 1, "offset": 0})
    keys(norm, ["multiplier", "offset"])
    multiplier = number(norm.get("multiplier", 1), "normalization multiplier", 1e-300)
    offset = number(norm.get("offset", 0), "normalization offset", -float("inf"))
    if entry["value_type"] == "category" and (multiplier != 1 or offset != 0):
        raise ValueError(
            "Set category weights explicitly instead of normalizing categorical inputs"
        )
    # A file is one extraction configuration; mixed run/request identifiers are never pooled.
    request_column = entry.get("request_id_column")
    if request_column and request_column not in frame:
        raise ValueError(f"Missing request identity column {request_column}")
    for name in dict.fromkeys(
        [request_column, "run_id", "configuration_id", "request_id", "model", "prompt_style"]
    ):
        if name and name in frame:
            identities = {str(v).strip() for v in frame[name]}
            if len(identities) > 1:
                raise ValueError(
                    f"Mixed request/configuration identities in {entry['id']} ({name})"
                )
            if (
                name == request_column
                and "request_id" in entry
                and identities != {str(entry["request_id"])}
            ):
                raise ValueError(f"Unexpected request identity in {entry['id']}")
    weights, categories, invalid, notes = {}, {}, {}, {}
    for row in frame.to_dict("records"):
        patient = row[entry["id_column"]]
        try:
            for name in ("success", "successful", "status", "classification_status"):
                if name in row:
                    allowed = (
                        {"true", "1", "1.0"}
                        if name in {"success", "successful"}
                        else {"success", "successful", "completed", "validated", "ok"}
                    )
                    if str(row[name]).strip().lower() not in allowed:
                        raise ValueError(f"Failed inference ({name})")
            if entry["value_type"] == "category":
                raw = str(row[column]).strip()
                cat = scoring["category_aliases"].get(raw, raw)
                if cat not in scoring["category_weights"]:
                    raise ValueError(f"Unknown category {raw!r}")
                score = scoring["category_weights"][cat]
                categories[patient] = cat
            else:
                score = number(row[column], f"score for {patient}") * multiplier + offset
            score = number(score, f"normalized score for {patient}")
            bounds = scoring["bounds"]
            if bounds and not bounds[0] <= score <= bounds[1]:
                raise ValueError("Score outside declared common scale bounds")
            weights[patient] = score
            notes[patient] = {
                k: str(row[k]).strip()
                for k in ("input_sha256", "sp_text_sha256", "assessment_text_sha256")
                if k in row and str(row[k]).strip()
            }
        except ValueError as exc:
            invalid[patient] = str(exc)
            categories.pop(patient, None)
    canonical = {
        "weights": weights,
        "categories": categories,
        "invalid": invalid,
        "notes": notes,
        "normalization": norm,
        "scale": scoring["scale"],
    }
    return {
        "entry": entry,
        **canonical,
        "sha256": file_hash(entry["path"]),
        "content_id": digest(canonical),
        "score_fingerprint": digest(weights),
    }


def choose(values, selection, name):
    if selection is None:
        return dict(sorted(values.items()))
    if not isinstance(selection, list):
        raise ValueError(f"{name} must be a list")
    unique(selection, name)
    missing = set(selection) - set(values)
    if missing:
        raise ValueError(f"Unknown {name}: {sorted(missing)}")
    return {k: values[k] for k in sorted(selection)}


def load_scenarios(path, base, scoring, models):
    if not path:
        return {}
    path = Path(path_at(path, base))
    value = read_yaml(path)
    schema(value, ["uncertainty"], ["uncertainty"])
    result = {}
    for mid, item in value["uncertainty"].items():
        keys(item, ["files"], ["files"])
        for entry in item["files"]:
            keys(entry, INPUT_KEYS, ["id", "path", "id_column", "value_column", "value_type"])
        unique([normalize_id(e["id"]) for e in item["files"]], "scenario IDs")
        if mid in models:
            result[mid] = [load_score(e, path.parent, scoring) for e in item["files"]]
    return result


def prepare(args):
    base = args.config.resolve().parent
    raw = read_yaml(args.config)
    schema(
        raw,
        ["chunks", "scores", "benchmark", "profiles", *DEFAULTS],
        ["chunks", "scores", "benchmark", "profiles"],
    )
    options = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    for key in DEFAULTS:
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    for key in (
        "chunks",
        "profiles",
        "output",
        "cache_dir",
        "uncertainty_manifest",
        "stress_test_manifest",
    ):
        cli = getattr(args, key, None)
        value = cli if cli is not None else raw.get(key, options.get(key))
        options[key] = path_at(value, Path.cwd() if cli is not None else base) if value else None
    for key, allowed in {
        "objectives": {"complexity-range", "lexicographic"},
        "selection": {"random", "matched", "robust"},
    }.items():
        if not isinstance(options[key], list):
            raise ValueError(f"{key} must be a list")
        unique(options[key], key)
        if set(options[key]) - allowed:
            raise ValueError(f"Invalid {key}")
        options[key] = sorted(options[key])
    for key, allowed in {
        "coverage": {"strict", "intersection"},
        "regret_bounds": {"none", "exact"},
        "uncertainty_source": {"files", "perturbations"},
    }.items():
        if options[key] not in allowed:
            raise ValueError(f"Invalid {key}")
    for key in ("samples", "robust_samples", "uncertainty_samples", "threads"):
        options[key] = integer(options[key], key, 1)
    for key in ("sample_seed", "uncertainty_seed", "solver_seed"):
        options[key] = integer(options[key], key)
    if options["solver_seed"] > 2_000_000_000:
        raise ValueError("solver_seed must be <= 2000000000")
    for key in ("perturb_fraction", "nominal_slack", "mip_gap", "optimality_tolerance"):
        options[key] = number(options[key], key)
    if options["perturb_fraction"] > 1 or not 1e-9 <= options["optimality_tolerance"] < 0.1:
        raise ValueError("fraction must be <= 1; optimality_tolerance must be in [1e-9, 0.1)")
    if options["time_limit"] is not None:
        options["time_limit"] = number(options["time_limit"], "time_limit", 1e-9)
    for key in ("resume", "dry_run", "allow_benchmark_in_uncertainty"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be boolean")
    scores = raw["scores"]
    keys(
        scores,
        ["inputs", "category_weights", "category_aliases", "scale", "score_min", "score_max"],
        ["inputs"],
    )
    weights = scores.get("category_weights", WEIGHTS)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("category_weights must be a nonempty mapping")
    weights = {str(k): number(v, "category weight") for k, v in weights.items()}
    aliases = scores.get("category_aliases", {})
    if not isinstance(aliases, dict) or set(aliases.values()) - set(weights):
        raise ValueError("Category aliases must map to explicit category weights")
    # Default repository score convention is 1--3, even for numeric registries.
    lo = number(scores.get("score_min", min(weights.values())), "score_min")
    hi = number(scores.get("score_max", max(weights.values())), "score_max")
    if hi < lo:
        raise ValueError("score_max must be >= score_min")
    scoring = {
        "category_weights": weights,
        "category_aliases": aliases,
        "scale": scores.get("scale", "repository additive complexity"),
        "bounds": [lo, hi],
    }
    for entry in scores["inputs"]:
        keys(entry, INPUT_KEYS, ["id", "path", "id_column", "value_column", "value_type"])
    entries = {normalize_id(e["id"]): e for e in scores["inputs"]}
    unique([normalize_id(e["id"]) for e in scores["inputs"]], "input IDs")
    benchmark = dict(raw["benchmark"])
    keys(benchmark, ["id", "kind", "provenance"], ["id"])
    if args.benchmark is not None:
        benchmark["id"] = args.benchmark
    benchmark["kind"] = args.benchmark_kind or benchmark.get("kind", "proxy")
    if benchmark["kind"] not in {"proxy", "clinical"} or benchmark["id"] not in entries:
        raise ValueError("Unknown benchmark or benchmark kind")
    if benchmark["kind"] == "clinical":
        provenance = entries[benchmark["id"]].get("provenance", {})
        keys(provenance, ["kind", "source", "description"], ["kind", "source", "description"])
        if (
            provenance["kind"] != "human"
            or not provenance["source"]
            or not provenance["description"]
        ):
            raise ValueError("Clinical benchmark requires human provenance on its registered input")
        benchmark["provenance"] = provenance
    models = choose(entries, options["models"], "models")
    options["models"] = sorted(models)
    inputs = {
        k: load_score(entries[k], base, scoring) for k in sorted(set(models) | {benchmark["id"]})
    }
    profiles = choose(
        load_profiles(options["profiles"], scoring["bounds"]), options["profile_ids"], "profile IDs"
    )
    chunk_path = Path(options["chunks"])
    if chunk_path.suffix.lower() == ".json":
        manifest = json.loads(chunk_path.read_text(encoding="utf-8"))
        # Accept a small explicit pointer manifest, or the existing chunk_config.json.
        if "membership" in manifest:
            schema(manifest, ["membership", "membership_sha256"], ["membership"])
            chunk_path = Path(path_at(manifest["membership"], chunk_path.parent))
        elif "membership_sha256" in manifest and "requested_size" in manifest:
            chunk_path = chunk_path.parent / "chunks.csv"
        else:
            raise ValueError("Chunk JSON must reference membership CSV using 'membership'")
        if (
            manifest.get("membership_sha256")
            and file_hash(chunk_path) != manifest["membership_sha256"]
        ):
            raise ValueError("Chunk manifest membership hash mismatch")
    chunks = choose(load_membership(chunk_path), options["chunk_ids"], "chunk IDs")
    construction = load_scenarios(options["uncertainty_manifest"], base, scoring, models)
    heldout = load_scenarios(options["stress_test_manifest"], base, scoring, models)
    missing_uncertainty = []
    if "robust" in options["selection"] and options["uncertainty_source"] == "files":
        missing_uncertainty = sorted(set(models) - set(construction))
        if missing_uncertainty and not options["dry_run"]:
            raise ValueError(f"Missing uncertainty files for models: {missing_uncertainty}")
    if options["uncertainty_source"] == "perturbations":
        if construction:
            raise ValueError("Do not combine perturbations with a construction-file manifest")
        if any(inputs[k]["entry"]["value_type"] != "category" for k in models):
            raise ValueError(
                "Categorical perturbations require categorical models; use files for numeric scores"
            )
        if len(set(weights.values())) != len(weights) or len(weights) < 2:
            raise ValueError("Perturbations require at least two strictly ordered category weights")
    oracle = set()
    reference = inputs[benchmark["id"]]
    # Human references and the active benchmark are prohibited, including byte/content aliases.
    references = [reference]
    for k, e in entries.items():
        if e.get("provenance", {}).get("kind") == "human" and k != benchmark["id"]:
            references.append(load_score(e, base, scoring))
    for mid, scenarios in construction.items():
        for scenario in scenarios:
            if any(
                scenario["entry"]["path"] == r["entry"]["path"]
                or scenario["sha256"] == r["sha256"]
                or scenario["content_id"] == r["content_id"]
                or scenario["score_fingerprint"] == r["score_fingerprint"]
                for r in references
            ):
                if not options["allow_benchmark_in_uncertainty"]:
                    raise ValueError(
                        f"Benchmark/reference leakage in uncertainty for {mid}: {scenario['entry']['id']}"
                    )
                oracle.add(mid)
    all_inputs = list(inputs.values()) + [
        s for group in (construction, heldout) for ss in group.values() for s in ss
    ]
    exclusions, effective = [], {}
    for chunk, patients in chunks.items():
        kept = []
        for patient in patients:
            problems = [
                f"{s['entry']['id']}: {s['invalid'].get(patient, 'missing ID')}"
                for s in all_inputs
                if patient not in s["weights"]
            ]
            if problems:
                if options["coverage"] == "strict":
                    raise ValueError(f"Coverage error {chunk}/{patient}: {'; '.join(problems)}")
                exclusions.append(
                    {"chunk_id": chunk, "patient_id": patient, "reason": "; ".join(problems)}
                )
            else:
                hashes = {}
                for s in all_inputs:
                    for key, value in s["notes"].get(patient, {}).items():
                        if key in hashes and hashes[key] != value:
                            raise ValueError(f"Note hash mismatch for {patient}: {key}")
                        hashes[key] = value
                kept.append(patient)
        effective[chunk] = kept
    instances = []
    for chunk, patients in effective.items():
        for pid, profile in profiles.items():
            cs = profile["clinicians"]
            capacity = sum(c["capacity"] - c["initial_count"] for c in cs)
            reason = (
                "empty effective chunk"
                if not patients
                else ("insufficient capacity" if capacity < len(patients) else "")
            )
            instances.append(
                {
                    "chunk_id": chunk,
                    "profile_id": pid,
                    "patient_count": len(patients),
                    "original_patient_count": len(chunks[chunk]),
                    "reduced": patients != chunks[chunk],
                    "patients": patients,
                    "clinicians": cs,
                    "feasible": not reason,
                    "reason": reason,
                    "initial_count_range": max(c["initial_count"] for c in cs)
                    - min(c["initial_count"] for c in cs),
                    "initial_load_range": max(c["initial_complexity"] for c in cs)
                    - min(c["initial_complexity"] for c in cs),
                }
            )
    return {
        "options": options,
        "scoring": scoring,
        "benchmark": benchmark,
        "inputs": inputs,
        "profiles": profiles,
        "chunks": effective,
        "instances": instances,
        "exclusions": exclusions,
        "construction": construction,
        "heldout": heldout,
        "oracle_models": sorted(oracle),
        "missing_uncertainty": missing_uncertainty,
        "source_hashes": {
            "config": file_hash(args.config),
            "profiles": file_hash(options["profiles"]),
            "chunks": file_hash(chunk_path),
        },
    }


def scenario_set(prepared, mid, instance, heldout=False):
    """Persisted unique vectors with multiplicities; nominal is always in construction set."""
    options, source = prepared["options"], prepared["inputs"][mid]
    patients = instance["patients"]
    nominal = {p: source["weights"][p] for p in patients}
    rows = [] if heldout else [{"id": "nominal", "weights": nominal, "changes": {}}]
    files = prepared["heldout" if heldout else "construction"].get(mid, [])
    for s in files:
        w = {p: s["weights"][p] for p in patients}
        rows.append(
            {
                "id": s["entry"]["id"],
                "weights": w,
                "sha256": s["sha256"],
                "path": s["entry"]["path"],
                "changes": {p: [nominal[p], w[p]] for p in patients if w[p] != nominal[p]},
            }
        )
    if not heldout and options["uncertainty_source"] == "perturbations":
        categories = sorted(
            prepared["scoring"]["category_weights"], key=prepared["scoring"]["category_weights"].get
        )
        count = math.ceil(options["perturb_fraction"] * len(patients))
        for index in range(options["uncertainty_samples"]):
            seed = seed_for(
                options["uncertainty_seed"], instance["chunk_id"], mid, source["content_id"], index
            )
            rng = random.Random(seed)
            w = dict(nominal)
            for p in rng.sample(patients, count):
                k = categories.index(source["categories"][p])
                neighbors = [j for j in (k - 1, k + 1) if 0 <= j < len(categories)]
                w[p] = prepared["scoring"]["category_weights"][categories[rng.choice(neighbors)]]
            rows.append(
                {
                    "id": f"perturbation_{index}",
                    "seed": seed,
                    "weights": w,
                    "changes": {p: [nominal[p], w[p]] for p in patients if w[p] != nominal[p]},
                }
            )
    unique_vectors = {}
    for row in rows:
        key = digest(row["weights"])
        if key not in unique_vectors:
            unique_vectors[key] = {
                "vector_id": key,
                "weights": row["weights"],
                "multiplicity": 0,
                "provenance": [],
            }
        unique_vectors[key]["multiplicity"] += 1
        unique_vectors[key]["provenance"].append({k: v for k, v in row.items() if k != "weights"})
    return sorted(unique_vectors.values(), key=lambda r: r["vector_id"])
