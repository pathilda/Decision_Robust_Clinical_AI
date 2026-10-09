"""Deterministic interrupted/error outcomes without timing-sensitive solves."""

from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from patient_assignment import cli, optimization
from patient_assignment.cohort import ID_COLUMN
from patient_assignment.inputs import parse_clinicians
from patient_assignment.optimization import Optimum, SolverDependencyError


def test_diagnostics_never_reads_missing_solution():
    class NoSolution:
        SolCount = 0
        Status = 9
        ObjBound = 0.0
        Runtime = 1.0

        @property
        def ObjVal(self):
            pytest.fail("ObjVal accessed without incumbent")

        @property
        def MIPGap(self):
            pytest.fail("MIPGap accessed without incumbent")

    names = (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "USER_OBJ_LIMIT",
    )
    gp = SimpleNamespace(GRB=SimpleNamespace(**{name: i + 1 for i, name in enumerate(names)}))
    result = optimization.stage_diagnostics(NoSolution(), gp)
    assert (
        result["status"] == "TIME_LIMIT" and result["objective"] is None and result["gap"] is None
    )


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_uncertified_stage1_never_moves_to_stage2(monkeypatch, has_incumbent):
    class Model:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def addVar(self, **kwargs):
            return 0

        def setObjective(self, *_):
            pass

        def optimize(self):
            self.calls += 1

    model = Model()
    session = SimpleNamespace(
        model=lambda: model, gp=SimpleNamespace(GRB=SimpleNamespace(MINIMIZE=1))
    )
    monkeypatch.setattr(optimization, "add_allocation", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(optimization, "constrain_range", lambda *args: None)
    monkeypatch.setattr(
        optimization,
        "stage_diagnostics",
        lambda *args: {"status": "TIME_LIMIT", "optimal": False, "has_incumbent": has_incumbent},
    )

    def extract(*args):
        assert has_incumbent, "solution extracted without incumbent"
        return {"p": "A"}

    monkeypatch.setattr(optimization, "extract_assignment", extract)
    clinicians = parse_clinicians([["A", "0", "0"]], 1)
    result = optimization.solve_lexicographic(session, ["p"], clinicians, {"p": 1}, 1)
    assert model.calls == 1 and not result.certified and "stage2" not in result.stages
    assert (result.assignment is not None) == has_incumbent


def inputs(tmp_path):
    chunks, a, b = [tmp_path / name for name in ("chunks.csv", "a.csv", "b.csv")]
    pd.DataFrame({"patient_id": ["p"], "chunk_id": ["c1"]}).to_csv(chunks, index=False)
    for path in (a, b):
        pd.DataFrame({ID_COLUMN: ["p"], "treatment_category": ["Easy to treatment"]}).to_csv(
            path, index=False
        )
    return [
        "compare-assignments",
        "--chunks",
        str(chunks),
        "--predictions-a",
        str(a),
        "--label-a",
        "AI",
        "--predictions-b",
        str(b),
        "--label-b",
        "Ref",
        "--clinician",
        "A",
        "0",
        "0",
        "--clinician-capacity",
        "1",
    ]


class FakeSession:
    def __init__(self, settings):
        self.gp = SimpleNamespace(
            GurobiError=RuntimeError, gurobi=SimpleNamespace(version=lambda: (0, 0, 0))
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def certified():
    stage = {
        "status": "OPTIMAL",
        "optimal": True,
        "has_incumbent": True,
        "objective": 0,
        "bound": 0,
        "gap": 0,
        "runtime": 0,
    }
    return Optimum({"stage1": stage, "stage2": stage}, {"p": "A"}, 0, Decimal(0))


@pytest.mark.parametrize("incumbent", [False, True])
def test_interrupted_matching_candidate_is_provisional(monkeypatch, tmp_path, incumbent):
    monkeypatch.setattr(cli, "SolverSession", FakeSession)
    monkeypatch.setattr(cli, "solve_lexicographic", lambda *args: certified())
    status = {
        "status": "TIME_LIMIT",
        "optimal": False,
        "has_incumbent": incumbent,
        "objective": 0 if incumbent else None,
        "bound": 0,
        "gap": 0 if incumbent else None,
        "runtime": 1,
    }
    pair = {"p": "A"} if incumbent else None
    monkeypatch.setattr(cli, "match_optima", lambda *args: (pair, pair, status))
    output = tmp_path / "output"
    assert cli.main(inputs(tmp_path) + ["--reference-side", "B", "--output", str(output)]) == 1
    results = pd.read_csv(output / "comparison_results.csv")
    assert not results.selection_certified.any()
    assert bool(results.provisional.iloc[0]) == incumbent
    allocations = pd.read_csv(output / "comparison_assignments.csv")
    assert len(allocations) == int(incumbent)
    if incumbent:
        assert results.raw_regret.iloc[0] == 0 and bool(results.regret_provisional.iloc[0])
    else:
        assert results.raw_regret.isna().all()
    assert results.minimum_disagreement_count.isna().all()


def test_uncertified_component_never_enters_matching(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "SolverSession", FakeSession)
    monkeypatch.setattr(
        cli,
        "solve_lexicographic",
        lambda *args: Optimum({"stage1": {"optimal": False, "status": "TIME_LIMIT"}}),
    )
    monkeypatch.setattr(
        cli, "match_optima", lambda *args: pytest.fail("Uncertified optimum used for matching")
    )
    assert cli.main(inputs(tmp_path) + ["--output", str(tmp_path / "out")]) == 1


def test_uncertified_random_selection_is_provisional_not_matching(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "SolverSession", FakeSession)
    monkeypatch.setattr(cli, "solve_lexicographic", lambda *args: certified())
    monkeypatch.setattr(
        cli, "match_optima", lambda *args: pytest.fail("Random mode called matching")
    )
    monkeypatch.setattr(
        cli,
        "random_optimum",
        lambda *args: (
            {"p": "A"},
            {
                "status": "TIME_LIMIT",
                "optimal": False,
                "has_incumbent": True,
                "objective": 0.1,
                "bound": 0,
                "gap": 1,
                "runtime": 1,
            },
        ),
    )
    output = tmp_path / "out"
    assert (
        cli.main(
            inputs(tmp_path)
            + ["--selection-mode", "random", "--selection-repeats", "2", "--output", str(output)]
        )
        == 1
    )
    rows = pd.read_csv(output / "comparison_results.csv")
    assert len(rows) == 2 and rows.provisional.all() and not rows.selection_certified.any()
    assert rows.matching_status.isna().all() and rows.raw_regret.isna().all()


def test_input_errors_no_outputs_and_dependency_failure_explicit(monkeypatch, tmp_path):
    arguments = inputs(tmp_path)
    output = tmp_path / "out"
    assert cli.main(arguments + ["--selection-repeats", "2", "--output", str(output)]) == 2
    assert not output.exists()

    def unavailable(settings):
        raise SolverDependencyError("Unavailable test license")

    monkeypatch.setattr(cli, "SolverSession", unavailable)
    assert cli.main(arguments + ["--output", str(output)]) == 2
    assert (output / "failure_status.json").is_file()
    assert not (output / "comparison_summary.json").exists()


def test_partial_random_incumbent_persisted_without_pair_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "SolverSession", FakeSession)
    monkeypatch.setattr(cli, "solve_lexicographic", lambda *args: certified())
    outcomes = iter(
        [
            ({"p": "A"}, {"status": "OPTIMAL", "optimal": True, "has_incumbent": True}),
            (None, {"status": "TIME_LIMIT", "optimal": False, "has_incumbent": False}),
        ]
    )
    monkeypatch.setattr(cli, "random_optimum", lambda *args: next(outcomes))
    output = tmp_path / "out"
    assert cli.main(inputs(tmp_path) + ["--selection-mode", "random", "--output", str(output)]) == 1
    partial = pd.read_csv(output / "selection_incumbents.csv")
    assert len(partial) == 1 and partial.side.iloc[0] == "A"
    results = pd.read_csv(output / "comparison_results.csv")
    assert results.observed_disagreement_count.isna().all()
    assert pd.read_csv(output / "comparison_assignments.csv").empty
