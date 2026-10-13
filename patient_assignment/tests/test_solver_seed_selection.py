"""Seed sensitivity uses original objectives and records every requested seed."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from patient_assignment import cli, selection
from patient_assignment.tests.test_failures import FakeSession, certified, inputs


def test_seeded_original_solve_sets_seed_and_keeps_session_unchanged(monkeypatch):
    models = []

    class Model:
        def setParam(self, key, value):
            self.seed = value
            assert key == "Seed"

    def model():
        result = Model()
        models.append(result)
        return result

    session = SimpleNamespace(gp=object(), model=model)
    optimum = certified()

    def solve(seeded, *args):
        assert seeded.gp is session.gp
        seeded.model()
        return optimum

    monkeypatch.setattr(selection, "solve_lexicographic", solve)
    monkeypatch.setattr(selection, "verify_optimal_set", lambda *args: None)
    assignment, status = selection.random_optimum(session, ["p"], [], {}, 1, optimum, 19)
    assert assignment == optimum.assignment and status["optimal"]
    assert models[0].seed == 19 and status["solver_seed"] == 19
    assert session.model is model


def test_nonoptimal_seeded_incumbent_not_exported(monkeypatch):
    optimum = certified()
    interrupted = certified()
    interrupted.stages["stage2"] = {
        "status": "TIME_LIMIT", "optimal": False, "has_incumbent": True,
    }
    monkeypatch.setattr(selection, "solve_lexicographic", lambda *args: interrupted)

    def reject(*args):
        raise ValueError("Not in certified optimal set")

    monkeypatch.setattr(selection, "verify_optimal_set", reject)
    assignment, status = selection.random_optimum(
        SimpleNamespace(gp=object()), ["p"], [], {}, 1, optimum, 0,
    )
    assert assignment is None and not status["optimal"]
    assert status["has_incumbent"]  # Diagnostic still reflects original solve.


def test_random_cli_seed_schedule_and_duplicate_assignments_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SolverSession", FakeSession)
    monkeypatch.setattr(cli, "solve_lexicographic", lambda *args: certified())
    calls = []

    def select(*args):
        calls.append(args[-1])
        return {"p": "A"}, {"status": "OPTIMAL", "optimal": True, "has_incumbent": True}

    monkeypatch.setattr(cli, "random_optimum", select)
    monkeypatch.setattr(cli, "match_optima", lambda *args: pytest.fail("Matching called"))
    output = tmp_path / "out"
    assert cli.main(inputs(tmp_path) + [
        "--selection-mode", "random", "--solver-seed", "10",
        "--selection-seed", "999", "--selection-repeats", "20", "--output", str(output),
    ]) == 0
    assert calls == [seed for seed in range(10, 30) for _ in range(2)]
    assignments = pd.read_csv(output / "comparison_assignments.csv")
    assert len(assignments) == 20 and assignments.solver_seed.tolist() == list(range(10, 30))
    assert assignments.selected_clinician_A.eq("A").all()
    streams = json.loads((output / "selection_streams.json").read_text())
    assert [row["solver_seed"] for row in streams] == calls
    config = json.loads((output / "comparison_config.json").read_text())
    assert config["selection_seed_active"] is False
    assert "original_objectives" in config["tie_policy"]


def test_seed_schedule_overflow_rejected_before_outputs(tmp_path):
    output = tmp_path / "out"
    assert cli.main(inputs(tmp_path) + [
        "--selection-mode", "random", "--solver-seed", "2000000000",
        "--selection-repeats", "2", "--output", str(output),
    ]) == 2
    assert not output.exists()
