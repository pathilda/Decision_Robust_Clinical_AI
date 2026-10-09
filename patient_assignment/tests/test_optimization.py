"""Real Gurobi tests checked against independently enumerated tiny optima."""

import itertools
import json
from decimal import Decimal

import pandas as pd
import pytest

from patient_assignment import cli
from patient_assignment.cohort import ID_COLUMN
from patient_assignment.inputs import parse_clinicians
from patient_assignment.optimization import (
    SolverDependencyError,
    SolverSession,
    SolverSettings,
    solve_lexicographic,
)
from patient_assignment.selection import (
    coefficients,
    match_optima,
    random_optimum,
    reference_regret,
)


@pytest.fixture
def session():
    pytest.importorskip(
        "gurobipy", reason="Gurobi Python API not installed; optimization unverified"
    )
    try:
        with SolverSession(SolverSettings()) as active:
            yield active
    except SolverDependencyError as exc:
        pytest.skip(f"Gurobi license unavailable: {exc}; optimization unverified")


def enumerate_optimal(patients, clinicians, weights, capacity):
    # Independent calculation: do not call production evaluator/formulation.
    candidates = []
    for choices in itertools.product([c.name for c in clinicians], repeat=len(patients)):
        assignment = dict(zip(patients, choices))
        counts = [c.initial_count + choices.count(c.name) for c in clinicians]
        if max(counts) > capacity:
            continue
        loads = [
            c.initial_severity + sum(weights[p] for p in patients if assignment[p] == c.name)
            for c in clinicians
        ]
        objective = (max(counts) - min(counts), max(loads) - min(loads))
        candidates.append((objective, assignment))
    best = min(value[0] for value in candidates)
    return best, [allocation for objective, allocation in candidates if objective == best]


@pytest.mark.parametrize(
    "initial,capacity,severity",
    [
        ([["A", "2", "0"], ["B", "0", "0"]], 3, [1, 3]),
        ([["A", "1", "3"], ["B", "0", "0"]], 2, [3]),
        ([["A", "0", "0"], ["B", "0", "0"]], 4, [3, 1, 1, 1]),
        ([["A", "0", "0.1"], ["B", "0", "0.333333333333333333"]], 3, [1, 2, 3]),
        ([["A", "1", "2"], ["B", "2", "3"], ["C", "0", "0"]], 3, [1, 3, 2]),
        ([["A", "1", "2.75"]], 4, [1, 3]),
    ],
)
def test_lexicographic_matches_exhaustive_enumeration(session, initial, capacity, severity):
    patients = [str(i) for i in range(len(severity))]
    clinicians = parse_clinicians(initial, capacity)
    weights = dict(zip(patients, severity))
    best, allocations = enumerate_optimal(patients, clinicians, weights, capacity)
    optimum = solve_lexicographic(session, patients, clinicians, weights, capacity)
    assert optimum.certified
    assert (optimum.count_range, optimum.severity_range) == best
    assert optimum.assignment in allocations
    if len(severity) == 4:
        # A 3/1 split can yield severity range zero, but worsens count range.
        assert best == (0, Decimal(2))
    if initial[0][1] == "2" and len(severity) == 2:
        assert set(optimum.assignment.values()) == {"B"}


@pytest.mark.parametrize(
    "weights_a,weights_b",
    [
        ([1, 1, 1, 1], [1, 1, 3, 3]),
        ([1, 2, 3, 1], [1, 2, 3, 1]),
        ([1, 1, 3, 3], [1, 3, 1, 3]),
    ],
)
def test_joint_matching_minimum_over_both_enumerated_sets(session, weights_a, weights_b):
    patients = list("1234")
    clinicians = parse_clinicians([["A", "0", "0"], ["B", "0", "0"]], 2)
    wa, wb = dict(zip(patients, weights_a)), dict(zip(patients, weights_b))
    oa, ob = [solve_lexicographic(session, patients, clinicians, w, 2) for w in (wa, wb)]
    _, sa = enumerate_optimal(patients, clinicians, wa, 2)
    _, sb = enumerate_optimal(patients, clinicians, wb, 2)
    true_min = min(sum(a[p] != b[p] for p in patients) for a in sa for b in sb)
    a, b, status = match_optima(session, patients, clinicians, wa, wb, 2, oa, ob)
    assert status["optimal"] and a in sa and b in sb
    assert sum(a[p] != b[p] for p in patients) == true_min
    if weights_a == weights_b:
        assert true_min == 0
    if weights_a == [1, 1, 1, 1]:
        assert reference_regret(a, patients, clinicians, wb, 2, ob)["raw_regret"] == "0"


def test_random_objective_preserves_optima_and_saved_regret(session):
    patients = list("1234")
    clinicians = parse_clinicians([["A", "0", "0"], ["B", "0", "0"]], 2)
    ai, ref = dict.fromkeys(patients, 1), dict(zip(patients, [1, 1, 3, 3]))
    oa, ob = [solve_lexicographic(session, patients, clinicians, w, 2) for w in (ai, ref)]
    _, allowed = enumerate_optimal(patients, clinicians, ai, 2)
    for repeat in range(1, 6):
        values, _ = coefficients(42, "chunk_0001", repeat, "AI", patients, ["A", "B"])
        assignment, status = random_optimum(session, patients, clinicians, ai, 2, oa, values)
        assert status["optimal"] and assignment in allowed
        assert reference_regret(assignment, patients, clinicians, ref, 2, ob)["raw_regret"] in {
            "0",
            "4",
        }


def command_inputs(tmp_path):
    membership = tmp_path / "chunks.csv"
    pd.DataFrame({"patient_id": list("12345"), "chunk_id": ["c1"] * 4 + ["c2"]}).to_csv(
        membership, index=False
    )
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame(
        {ID_COLUMN: list("12345"), "treatment_category": ["Easy to treatment"] * 5}
    ).to_csv(a, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: list("12345"),
            "treatment_category": ["Easy to treatment"] * 2
            + ["severe to treatment"] * 2
            + ["Easy to treatment"],
        }
    ).to_csv(b, index=False)
    return [
        "compare-assignments",
        "--chunks",
        str(membership),
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
        "--clinician",
        "B",
        "0",
        "0",
        "--clinician-capacity",
        "2",
    ]


def test_random_cli_no_matching_reference_independence_repeats_and_exports(
    session, tmp_path, monkeypatch
):
    args = command_inputs(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Random mode must not invoke joint matching")

    monkeypatch.setattr(cli, "match_optima", forbidden)
    common = args + ["--selection-mode", "random", "--selection-repeats", "3"]
    assert cli.main(common + ["--output", str(tmp_path / "no_reference")]) == 0
    assert (
        cli.main(common + ["--reference-side", "B", "--output", str(tmp_path / "reference")]) == 0
    )
    assignments = pd.read_csv(
        tmp_path / "reference/comparison_assignments.csv", dtype={"patient_id": str}
    )
    without = pd.read_csv(
        tmp_path / "no_reference/comparison_assignments.csv", dtype={"patient_id": str}
    )
    assert assignments.selected_clinician_A.tolist() == without.selected_clinician_A.tolist()
    assert len(assignments) == 15 and set(assignments.repeat_id) == {1, 2, 3}
    results = pd.read_csv(tmp_path / "reference/comparison_results.csv")
    assert (
        len(results) == 6
        and results.matching_status.isna().all()
        and results.minimum_disagreement_count.isna().all()
    )
    assert pd.read_csv(tmp_path / "no_reference/comparison_results.csv").raw_regret.isna().all()
    for row in results.to_dict("records"):
        selected = assignments.loc[
            (assignments.chunk_id == row["chunk_id"]) & (assignments.repeat_id == row["repeat_id"])
        ]
        assert (
            sum(selected.selected_clinician_A != selected.selected_clinician_B)
            == row["observed_disagreement_count"]
        )
        loads = {
            name: sum(selected.loc[selected.selected_clinician_A == name, "weight_B"])
            for name in ("A", "B")
        }
        assert max(loads.values()) - min(loads.values()) - row["G_reference"] == row["raw_regret"]
    # A/B swaps and partner labels do not affect a model's random coefficients or selection.
    swapped = args.copy()
    for option_a, option_b in (("--predictions-a", "--predictions-b"), ("--label-a", "--label-b")):
        ia, ib = swapped.index(option_a) + 1, swapped.index(option_b) + 1
        swapped[ia], swapped[ib] = swapped[ib], swapped[ia]
    assert (
        cli.main(
            swapped
            + [
                "--selection-mode",
                "random",
                "--selection-repeats",
                "3",
                "--reference-side",
                "A",
                "--output",
                str(tmp_path / "swap"),
            ]
        )
        == 0
    )
    swap = pd.read_csv(tmp_path / "swap/comparison_assignments.csv")
    assert swap.selected_clinician_B.tolist() == assignments.selected_clinician_A.tolist()
    summary = json.loads((tmp_path / "reference/comparison_summary.json").read_text())
    assert summary["successful_chunks"] == 2 and summary["certified_selections"] == 6


def test_matched_cli_exported_pair_reproduces_regret(session, tmp_path):
    args = command_inputs(tmp_path)
    assert cli.main(args + ["--reference-side", "B", "--output", str(tmp_path / "matched")]) == 0
    results = pd.read_csv(tmp_path / "matched/comparison_results.csv")
    assert (results.raw_regret == 0).all()
    assignments = pd.read_csv(tmp_path / "matched/comparison_assignments.csv")
    assert not assignments.changed.any()
    config = json.loads((tmp_path / "matched/comparison_config.json").read_text())
    assert config["tie_policy"] == "solver_selected_closest_pair"
    assert config["selection_seed_active"] is False


def test_solve_cli_remainder_and_decimal_output(session, tmp_path):
    args = command_inputs(tmp_path)
    membership, predictions = (
        args[args.index("--chunks") + 1],
        args[args.index("--predictions-a") + 1],
    )
    assert (
        cli.main(
            [
                "solve",
                "--chunks",
                membership,
                "--predictions",
                predictions,
                "--label",
                "AI",
                "--clinician",
                "A",
                "0",
                "0.1234567890123456789",
                "--clinician",
                "B",
                "0",
                "0",
                "--clinician-capacity",
                "2",
                "--output",
                str(tmp_path / "solve"),
            ]
        )
        == 0
    )
    results = pd.read_csv(tmp_path / "solve/chunk_results.csv")
    assert list(results.patient_count) == [4, 1] and results.component_optimal.all()
    with (tmp_path / "solve/clinician_loads.csv").open() as handle:
        assert "0.1234567890123456789" in handle.read()
