"""Registry/profile validation, exact coverage, stable identities and leakage checks."""

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from patient_assignment.cli import build_parser
from patient_assignment.experiment import ArtifactCache, run_experiment
from patient_assignment.experiment_inputs import generate_profiles, prepare, read_yaml, scenario_set

DEMO = Path(__file__).resolve().parents[2] / "configs" / "assignment_demo"


@pytest.fixture
def config(tmp_path):
    target = tmp_path / "inputs"
    shutil.copytree(DEMO, target)
    return target / "experiment.yaml"


def prepared(config, *flags):
    return prepare(
        build_parser().parse_args(["assignment-experiment", "--config", str(config), *flags])
    )


def update(config, transform):
    value = read_yaml(config)
    transform(value)
    config.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_override_lists_replace_and_unknown_keys_rejected(config):
    p = prepared(config, "--models", "synthetic_ai", "--samples", "2")
    assert p["options"]["models"] == ["synthetic_ai"] and p["options"]["samples"] == 2
    update(config, lambda v: v.update(typo=True))
    with pytest.raises(ValueError, match="Unknown keys"):
        prepared(config)


def test_duplicate_yaml_keys_and_ids_rejected(config):
    with config.open("a") as f:
        f.write("\nsamples: 4\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        prepared(config)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (
            lambda v: v["scores"]["inputs"].append(copy.deepcopy(v["scores"]["inputs"][0])),
            "unique IDs",
        ),
        (lambda v: v["scores"]["inputs"][0].update(scale="other scale"), "Incompatible scale"),
        (lambda v: v["benchmark"].update(kind="clinical"), "Missing keys"),
        (lambda v: v.update(samples=0), "must be finite"),
        (lambda v: v.update(objectives=["not-a-mode"]), "Invalid objectives"),
        (lambda v: v.update(optimality_tolerance=0.5), "optimality_tolerance"),
    ],
)
def test_invalid_config(config, mutation, pattern):
    update(config, mutation)
    with pytest.raises(ValueError, match=pattern):
        prepared(config)


def test_missing_invalid_and_duplicate_predictions(config):
    path = config.parent / "ai.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Coverage error"):
        prepared(config)
    p = prepared(config, "--coverage", "intersection")
    assert "001" not in p["chunks"]["chunk_01"]
    assert p["exclusions"][0]["patient_id"] == "001"
    frame.loc[0, "category"] = "made up category"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown category"):
        prepared(config)
    assert len(prepared(config, "--coverage", "intersection")["chunks"]["chunk_01"]) == 3
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate normalized IDs"):
        prepared(config, "--coverage", "intersection")


def test_failed_records_numeric_scores_and_mixed_requests(config):
    path = config.parent / "anchor.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["status"] = "failed"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Failed inference"):
        prepared(config)
    frame["status"] = "ok"
    frame.loc[0, "score"] = "NaN"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        prepared(config)
    frame.loc[0, "score"] = "1.2"
    frame["model"] = ["m1"] * 4 + ["m2"] * 4
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Mixed request"):
        prepared(config)


def test_leakage_by_identity_copy_and_reordered_content(config):
    uncertainty = config.parent / "uncertainty.yaml"
    data = read_yaml(uncertainty)
    data["uncertainty"]["synthetic_ai"]["files"][0]["path"] = "proxy_copy.csv"
    uncertainty.write_text(yaml.safe_dump(data), encoding="utf-8")
    proxy = pd.read_csv(config.parent / "proxy.csv", dtype=str)
    proxy.iloc[::-1].to_csv(config.parent / "proxy_copy.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        prepared(config)
    p = prepared(config, "--allow-benchmark-in-uncertainty")
    assert p["oracle_models"] == ["synthetic_ai"]


def test_perturbations_exact_changes_dedup_and_numeric_rejection(config):
    update(config, lambda v: v.update(uncertainty_manifest=None, stress_test_manifest=None))
    with pytest.raises(ValueError, match="categorical models"):
        prepared(config, "--uncertainty-source", "perturbations")
    p = prepared(
        config,
        "--models",
        "synthetic_ai",
        "--uncertainty-source",
        "perturbations",
        "--perturb-fraction",
        "0.1",
        "--uncertainty-samples",
        "10",
    )
    vectors = scenario_set(p, "synthetic_ai", p["instances"][0])
    assert sum(v["multiplicity"] for v in vectors) == 11
    for vector in vectors:
        for source in vector["provenance"]:
            if source["id"] != "nominal":
                assert len(source["changes"]) == 1  # ceil(0.1 * 4)
                assert all(abs(a - b) == 1 for a, b in source["changes"].values())


def test_profiles_conservation_bounds_permutation_and_impossibility():
    args = SimpleNamespace(
        clinician=[["A", "10", "20"], ["B", "10", "20"], ["C", "10", "20"]],
        clinician_capacity=25,
        score_min=1,
        score_max=3,
        imbalance_factors=[0, 0.5, 1],
        profile_seed=42,
    )
    profiles = generate_profiles(args)
    assert [p["metadata"]["initial_load_range"] for p in profiles["profiles"]] == [0, 10, 20]
    for p in profiles["profiles"]:
        assert sum(c["initial_complexity"] for c in p["clinicians"]) == pytest.approx(60)
        assert all(c["initial_count"] == 10 and c["capacity"] == 25 for c in p["clinicians"])
    args.clinician.reverse()
    assert generate_profiles(args) == profiles
    args.imbalance_factors = [1.1]
    with pytest.raises(ValueError, match="in \\[0,1\\]"):
        generate_profiles(args)
    args.imbalance_factors = [0, 1]
    args.clinician[0][1] = "11"
    with pytest.raises(ValueError, match="equal initial counts"):
        generate_profiles(args)
    args.clinician = [["A", "0", "0"]]
    assert generate_profiles(args)["metadata"]["identical_profiles"]


def test_infeasible_instances_and_empty_intersection_are_explicit(config):
    path = config.parent / "profiles.yaml"
    update(path, lambda v: [c.update(capacity=2) for p in v["profiles"] for c in p["clinicians"]])
    p = prepared(config)
    assert all(not i["feasible"] and i["reason"] == "insufficient capacity" for i in p["instances"])
    frame = pd.read_csv(config.parent / "ai.csv", dtype=str)
    frame.iloc[4:].to_csv(config.parent / "ai.csv", index=False)
    p = prepared(config, "--coverage", "intersection")
    assert p["chunks"]["chunk_01"] == []
    assert p["instances"][0]["reason"] == "empty effective chunk"


def test_dry_run_never_initializes_solver_and_reports_missing_uncertainty(
    config, monkeypatch, capsys
):
    from patient_assignment import optimization

    monkeypatch.setattr(optimization, "gurobi", lambda: pytest.fail("initialized Gurobi"))
    update(config, lambda v: v.update(uncertainty_manifest=None))
    p = prepared(config, "--dry-run")
    assert run_experiment(p) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["missing_uncertainty"] == ["synthetic_ai", "synthetic_anchor"]
    assert report["planned_solve_stages_before_cache"]["regret_bounds"] == 168


def test_row_and_model_order_keep_content_identity(config):
    a = prepared(config)
    for path in config.parent.glob("*.csv"):
        pd.read_csv(path, dtype=str).iloc[::-1].to_csv(path, index=False)
    b = prepared(config, "--models", "synthetic_anchor", "synthetic_ai")
    assert a["chunks"] == b["chunks"]
    assert {k: v["content_id"] for k, v in a["inputs"].items()} == {
        k: v["content_id"] for k, v in b["inputs"].items()
    }


def test_cache_rejects_stale_record_and_never_caches_failed_solve(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")

    def failed():
        raise ValueError("incomplete")

    with pytest.raises(ValueError):
        cache.get({"task": "f"}, failed)
    assert not list(cache.path.iterdir())
    cache.get({"task": "ok"}, lambda: {"value": 3})
    path = next(cache.path.iterdir())
    record = json.loads(path.read_text())
    record["payload"]["value"] = 4
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Stale/corrupt"):
        cache.get({"task": "ok"}, lambda: pytest.fail("stale result reused"))
