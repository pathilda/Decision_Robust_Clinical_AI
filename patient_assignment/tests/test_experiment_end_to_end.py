"""Complete table exports, benchmark independence, resume, and solver-free plots."""

import json
import shutil

import pandas as pd
import pytest

from patient_assignment import cli, experiment_solver, optimization
from patient_assignment.experiment import run_experiment
from patient_assignment.optimization import SolverDependencyError, SolverSession, SolverSettings
from patient_assignment.tests.test_experiment_inputs import DEMO, prepared


@pytest.fixture
def config(tmp_path):
    target = tmp_path / "inputs"
    shutil.copytree(DEMO, target)
    return target / "experiment.yaml"


@pytest.fixture
def license_available():
    try:
        with SolverSession(SolverSettings()):
            pass
    except SolverDependencyError as exc:
        pytest.skip(str(exc))


def run(config, output, *flags):
    p = prepared(
        config,
        "--samples",
        "4",
        "--profile-ids",
        "balanced",
        "unbalanced",
        "--output",
        str(output),
        *flags,
    )
    return run_experiment(p)


def test_full_run_replot_resume_and_benchmark_independence(
    config, tmp_path, license_available, monkeypatch
):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run(config, first) == 0
    status = json.loads((first / "run_status.json").read_text())
    assert status["complete"] and status["valid_selections"] == 96
    rows = pd.read_csv(first / "solutions.csv")
    bounds = pd.read_csv(first / "regret_bounds.csv")
    summary = pd.read_csv(first / "summaries.csv")
    assignments = pd.read_csv(first / "assignments.csv", dtype={"patient_id": str})
    assert set(assignments.patient_id) == {f"{p:03d}" for p in range(1, 9)}
    assert rows.certified.all() and bounds.complete.all()
    assert (rows.regret >= 0).all()
    assert rows.heldout_w.notna().all()
    assert summary.loc[summary.level == "across_chunks", "instance_count"].eq(2).all()
    assert (
        summary.loc[(summary.level == "instance") & (summary.strategy == "random"), "draws"]
        .eq(4)
        .all()
    )
    # Full execution with changed benchmark must keep random and robust candidate selections.
    assert run(config, second, "--benchmark", "synthetic_anchor") == 0
    other = pd.read_csv(second / "solutions.csv")
    keys = ["model_id", "chunk_id", "profile_id", "objective", "strategy", "sample"]
    independent = rows.loc[rows.strategy != "matched"].sort_values(keys)
    switched = other.loc[other.strategy != "matched"].sort_values(keys)
    assert independent.assignment_hash.tolist() == switched.assignment_hash.tolist()
    assert (
        other.loc[(other.model_id == "synthetic_anchor") & (other.nominal_slack == 0), "regret"]
        .le(1e-6)
        .all()
    )
    assert json.loads((second / "run_status.json").read_text())["cache_hits"] > 0
    # Reordered source CSV rows give identical independently selected allocations.
    for path in config.parent.glob("*.csv"):
        pd.read_csv(path, dtype=str).iloc[::-1].to_csv(path, index=False)
    third = tmp_path / "reordered"
    assert run(config, third, "--models", "synthetic_anchor", "synthetic_ai") == 0
    reordered = pd.read_csv(third / "solutions.csv")
    assert (
        independent.assignment_hash.tolist()
        == reordered.loc[reordered.strategy != "matched"].sort_values(keys).assignment_hash.tolist()
    )
    # Plot-only and completed resume must work when solver import/initialization is forbidden.
    monkeypatch.setattr(
        optimization, "gurobi", lambda: pytest.fail("plot/resume initialized solver")
    )
    assert run(config, third, "--resume", "--models", "synthetic_anchor", "synthetic_ai") == 0
    assert (
        cli.main(
            [
                "assignment-plots",
                "--results",
                str(first),
                "--output",
                str(tmp_path / "plots"),
                "--format",
                "png",
                "pdf",
                "--max-instances-per-figure",
                "2",
            ]
        )
        == 0
    )
    assert list((tmp_path / "plots").glob("*.png")) and list((tmp_path / "plots").glob("*.pdf"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(config, third, "--resume", "--samples", "5")
    (third / "solutions.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(config, third, "--resume", "--models", "synthetic_anchor", "synthetic_ai")


def test_failed_nominal_has_no_fallback_or_silent_metrics(
    config, tmp_path, license_available, monkeypatch
):
    def incomplete(*args):
        raise experiment_solver.IncompleteSolve("nominal", {"status": "TIME_LIMIT", "bound": 0})

    monkeypatch.setattr(experiment_solver, "nominal", incomplete)
    output = tmp_path / "failed"
    assert run(config, output) == 1
    rows = pd.read_csv(output / "solutions.csv")
    assert not rows.certified.any() and rows.regret.isna().all()
    assert pd.read_csv(output / "assignments.csv").empty
    assert not json.loads((output / "run_status.json").read_text())["complete"]
    assert not list((tmp_path / ".assignment_cache").glob("*.json"))


def test_oracle_excluded_from_deployable_summary(config, tmp_path, license_available):
    import yaml

    path = config.parent / "uncertainty.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["uncertainty"]["synthetic_ai"]["files"][0]["path"] = "proxy.csv"
    path.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "oracle"
    assert run(config, output, "--allow-benchmark-in-uncertainty") == 0
    summaries = pd.read_csv(output / "deployable_summaries.csv")
    assert "synthetic_ai" not in set(summaries.model_id)
    assert "matched" not in set(summaries.strategy)
