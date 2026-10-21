"""Strict probability inputs and versioned uncertainty experiment grids (no solver)."""

# Validation errors deliberately use ValueError at the CLI boundary.
# ruff: noqa: TRY004
import itertools
import math
from pathlib import Path

from .cohort import normalize_id
from .experiment_inputs import digest, integer, keys, number, read_yaml, unique
from .inputs import file_hash, read_table

DEFAULTS = {
    "schema_version": 1,
    "probabilities": {
        "columns": {
            "easy": "probability_easy",
            "moderate": "probability_moderate",
            "severe": "probability_severe",
        },
        "sum_tolerance": 0.0001,
        "renormalize": False,
        "bad_row": "error",
        "point_prediction_consistency": "warn",
        "temperature_values": [1.0],
    },
    "stochastic": {
        "scenario_counts": [20, 50, 100, 200],
        "scenario_seeds": [42, 123, 2026],
        "method": "categorical_iid",
        "probabilities": "uniform_empirical",
    },
    "robust": {
        "uncertainty_type": "budgeted_discrete",
        "gamma_fractions": [0.0, 0.1, 0.25, 0.5, 1.0],
        "gamma_counts": [],
        "gamma_rounding": "ceil",
        "support_policy": "probability_threshold",
        "probability_thresholds": [0.05, 0.2],
        "ensure_nominal_in_support": True,
    },
    "evaluation": {
        "benchmark": "fixed_proxy_point",
        "holdout_scenarios": 500,
        "holdout_seed": 7781,
        "cvar_alpha": 0.9,
    },
    "plots": {
        "include_policy_comparisons": True,
        "include_parameter_sweeps": True,
        "include_assignment_comparisons": True,
    },
}


def configuration(path, policies, requested=None):
    raw = read_yaml(path) if path else {"schema_version": 1}
    keys(raw, DEFAULTS)
    if raw.get("schema_version") != 1:
        raise ValueError("uncertainty schema_version must be 1")
    config = {"schema_version": 1}
    for section, defaults in DEFAULTS.items():
        if section == "schema_version":
            continue
        keys(raw.get(section, {}), defaults)
        config[section] = dict(defaults, **raw.get(section, {}))
    fixed = {
        "probabilities": {"bad_row": "error"},
        "stochastic": {"method": "categorical_iid", "probabilities": "uniform_empirical"},
        "robust": {
            "uncertainty_type": "budgeted_discrete",
            "gamma_rounding": "ceil",
            "support_policy": "probability_threshold",
            "ensure_nominal_in_support": True,
        },
        "evaluation": {"benchmark": "fixed_proxy_point"},
    }
    for section, values in fixed.items():
        for key, value in values.items():
            if config[section][key] != value:
                raise ValueError(f"{section}.{key}: only {value!r} is supported")
    p, st, r, ev = (config[k] for k in ("probabilities", "stochastic", "robust", "evaluation"))
    if p["point_prediction_consistency"] not in ("warn", "error"):
        raise ValueError("point_prediction_consistency must be warn or error")
    if not isinstance(p["renormalize"], bool):
        raise ValueError("renormalize must be boolean")
    p["sum_tolerance"] = number(p["sum_tolerance"], "sum_tolerance")
    if p["sum_tolerance"] > 0.01:
        raise ValueError("sum_tolerance must be <= .01")
    for section, field, minimum, integral in (
        (p, "temperature_values", 1e-12, False),
        (st, "scenario_counts", 1, True),
        (st, "scenario_seeds", 0, True),
        (r, "gamma_fractions", 0, False),
        (r, "gamma_counts", 0, True),
        (r, "probability_thresholds", 0, False),
    ):
        values = section[field]
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        if field != "gamma_counts" or values:
            unique(values, field)
        section[field] = [(integer if integral else number)(v, field, minimum) for v in values]
    if any(v > 1 for v in r["gamma_fractions"] + r["probability_thresholds"]):
        raise ValueError("Gamma fractions and probability thresholds must be in [0,1]")
    ev["holdout_scenarios"] = integer(ev["holdout_scenarios"], "holdout_scenarios")
    ev["holdout_seed"] = integer(ev["holdout_seed"], "holdout_seed")
    ev["cvar_alpha"] = number(ev["cvar_alpha"], "cvar_alpha")
    if ev["cvar_alpha"] >= 1:
        raise ValueError("cvar_alpha must be < 1")
    for value in config["plots"].values():
        if not isinstance(value, bool):
            raise ValueError("plot settings must be boolean")
    unique(policies, "policies")
    grid = []
    if "deterministic" in policies:
        grid.append({"policy": "deterministic"})
    if "stochastic" in policies:
        grid.extend(
            {"policy": "stochastic", "scenario_count": n, "scenario_seed": seed, "temperature": t}
            for n, seed, t in itertools.product(
                st["scenario_counts"], st["scenario_seeds"], p["temperature_values"]
            )
        )
    if "robust" in policies:
        budgets = (
            [{"gamma_count": n} for n in r["gamma_counts"]]
            if r["gamma_counts"]
            else [{"gamma_fraction": v} for v in r["gamma_fractions"]]
        )
        grid.extend(
            dict(policy="robust", probability_threshold=t, **g)
            for g, t in itertools.product(budgets, r["probability_thresholds"])
        )
    for cell in grid:
        cell["policy_configuration_id"] = cell["policy"] + "-" + digest(cell)[:12]
    if requested:
        unknown = set(requested) - {c["policy_configuration_id"] for c in grid}
        if unknown:
            raise ValueError(f"Unknown policy configuration IDs: {sorted(unknown)}")
        grid = [c for c in grid if c["policy_configuration_id"] in requested]
    return config, grid


def load_probabilities(args, prepared, config):
    entries = {normalize_id(e["id"]): e for e in read_yaml(args.config)["inputs"]}
    order = prepared["resolved"]["category_order"]
    patients = sorted(p for ps in prepared["resolved"]["chunks"].values() for p in ps)
    settings = config["probabilities"]
    result, diagnostics, sources = {}, [], {}
    for mid in args.models:
        entry = entries[mid]
        path = Path(entry.get("probability_path", entry["path"]))
        path = (args.config.resolve().parent / path).resolve()
        columns = entry.get("probability_columns", settings["columns"])
        aliases = prepared["resolved"]["category_aliases"]
        if not isinstance(columns, dict) or len(columns) != len(order):
            raise ValueError(f"{mid}: probability_columns must have exactly one entry per category")
        columns = {aliases.get(k, k): v for k, v in columns.items()}
        if set(columns) != set(order) or len(set(columns.values())) != len(order):
            raise ValueError(f"{mid}: probability_columns must uniquely map category_order")
        id_column = entry.get("probability_id_column", entry["id_column"])
        table = read_table(path, entry.get("probability_sheet", entry.get("sheet")))
        missing = {id_column, *columns.values()} - set(table.columns)
        if missing:
            raise ValueError(
                f"{mid}: missing probability columns {sorted(missing)}; supply a full "
                "probability vector or probability_path; scalar confidence is insufficient"
            )
        rows = {}
        for row in table.to_dict("records"):
            pid = normalize_id(row[id_column])
            if pid in rows:
                raise ValueError(f"{mid}: duplicate probability patient ID {pid}")
            rows[pid] = row
        if set(patients) - set(rows):
            raise ValueError(
                f"{mid}: probability input missing {len(set(patients) - set(rows))} chunk patients"
            )
        data, corrections, disagreements = {}, 0, 0
        for pid in patients:
            values = [number(rows[pid][columns[c]], f"{mid}: probability {c}") for c in order]
            total = math.fsum(values)
            if max(values) > 1 or total <= 0 or abs(total - 1) > settings["sum_tolerance"]:
                raise ValueError(f"{mid}: invalid probability vector for de-identified ID {pid}")
            if abs(total - 1) > 1e-12:
                if not settings["renormalize"]:
                    raise ValueError(
                        f"{mid}: probabilities do not sum to 1; explicitly enable "
                        "probabilities.renormalize to correct within sum_tolerance"
                    )
                values = [v / total for v in values]
                corrections += 1
            disagreement = (
                order[max(range(len(order)), key=lambda k: values[k])]
                != prepared["loaded"][mid]["categories"][pid]
            )
            disagreements += disagreement
            if disagreement and settings["point_prediction_consistency"] == "error":
                raise ValueError(f"{mid}: point label disagrees with probability argmax")
            data[pid] = values
        result[mid] = data
        sources[mid] = {
            "path": str(path),
            "sha256": file_hash(path),
            "normalized_sha256": digest(data),
            "columns": columns,
            "id_column": id_column,
        }
        diagnostics.append(
            {
                "model_id": mid,
                "n_patients": len(data),
                "valid": True,
                "renormalized_rows": corrections,
                "point_argmax_disagreements": disagreements,
                "warning": "Point labels preserved; disagreement is not a clinical accuracy error",
                **sources[mid],
            }
        )
    return result, diagnostics, sources
