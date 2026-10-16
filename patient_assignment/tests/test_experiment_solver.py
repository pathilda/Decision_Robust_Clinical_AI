"""Independent exhaustive oracles for the new experiment's optimization semantics."""

import itertools
from types import SimpleNamespace

import pytest

from patient_assignment import experiment_solver as solver
from patient_assignment.optimization import SolverDependencyError, SolverSession


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("gurobipy")
    try:
        with SolverSession(solver.ExperimentSettings()) as session:
            yield session
    except SolverDependencyError as exc:
        pytest.skip(str(exc))


def instance(n=4, clinicians=None):
    return {
        "patients": [str(p) for p in range(n)],
        "chunk_id": "c",
        "profile_id": "p",
        "clinicians": clinicians
        or [
            {"id": j, "initial_count": 0, "initial_complexity": 0, "capacity": n}
            for j in ("A", "B")
        ],
    }


def arithmetic(inst, weights, chosen):
    counts, loads = [], []
    for c in inst["clinicians"]:
        assigned = [p for p in inst["patients"] if chosen[p] == c["id"]]
        counts.append(c["initial_count"] + len(assigned))
        loads.append(c["initial_complexity"] + sum(weights[p] for p in assigned))
        if counts[-1] > c["capacity"]:
            return None
    return max(counts) - min(counts), max(loads) - min(loads), sum(loads)


def enumerate_set(inst, weights, mode):
    possible = []
    for labels in itertools.product(
        [c["id"] for c in inst["clinicians"]], repeat=len(inst["patients"])
    ):
        chosen = dict(zip(inst["patients"], labels))
        values = arithmetic(inst, weights, chosen)
        if values is not None:
            objective = values[:2] if mode == "lexicographic" else (values[1],)
            possible.append((objective, chosen))
    best = min(o for o, _ in possible)
    return best, [a for o, a in possible if o == best]


@pytest.mark.parametrize("mode", ["complexity-range", "lexicographic"])
@pytest.mark.parametrize(
    "scores,reference,clinicians",
    [
        ([3, 1, 1, 1], [1, 1, 3, 3], None),
        ([1, 1, 1, 1], [1, 1, 3, 3], None),
        ([1, 1, 1, 1], [1, 1, 1, 1], None),
        ([0.1, 0.2, 0.3, 0.2], [0.2, 0.3, 0.1, 0.2], None),
        ([1, 2], [3, 1], [{"id": "A", "initial_count": 1, "initial_complexity": 2, "capacity": 4}]),
        (
            [1, 3, 2],
            [3, 1, 2],
            [
                {"id": "A", "initial_count": 2, "initial_complexity": 3, "capacity": 3},
                {"id": "B", "initial_count": 1, "initial_complexity": 1, "capacity": 4},
            ],
        ),
    ],
)
def test_all_selection_models_against_enumeration(session, mode, scores, reference, clinicians):
    inst = instance(len(scores), clinicians)
    weights, benchmark = dict(zip(inst["patients"], scores)), dict(zip(inst["patients"], reference))
    best, allowed = enumerate_set(inst, weights, mode)
    bench_best, bench_allowed = enumerate_set(inst, benchmark, mode)
    optimum = solver.nominal(session, inst, weights, mode, 1e-6)
    bench_optimum = solver.nominal(session, inst, benchmark, mode, 1e-6)
    assert optimum["g"] == pytest.approx(best[-1])
    if mode == "lexicographic":
        assert optimum["d"] == best[0]
    if scores == [3, 1, 1, 1]:
        assert optimum["g"] == (0 if mode == "complexity-range" else 2)
        assert arithmetic(inst, weights, optimum["assignment"])[0] == (
            2 if mode == "complexity-range" else 0
        )
    total = sum(scores) + sum(c["initial_complexity"] for c in inst["clinicians"])
    assert all(arithmetic(inst, weights, a)[2] == pytest.approx(total) for a in allowed)
    for seed in range(5):
        chosen = solver.random_selection(session, inst, weights, optimum, 1e-6, seed)["assignment"]
        assert any(chosen == a for a in allowed)
    matched = solver.matched_selection(
        session, inst, weights, benchmark, optimum, bench_optimum, 1e-6
    )
    distance = lambda a, b: sum(a[p] != b[p] for p in inst["patients"])
    assert distance(matched["assignment"], matched["benchmark_assignment"]) == min(
        distance(a, b) for a in allowed for b in bench_allowed
    )
    envelope = solver.regret_envelope(
        session, inst, weights, benchmark, optimum, bench_optimum, 1e-6
    )
    regrets = [arithmetic(inst, benchmark, a)[1] - bench_best[-1] for a in allowed]
    assert envelope["complete"]
    assert envelope["minimum_regret"] == pytest.approx(min(regrets))
    assert envelope["maximum_regret"] == pytest.approx(max(regrets))
    for witness in envelope["witnesses"].values():
        solver.verify(inst, weights, witness, optimum, 1e-6)
    vectors = [
        {"weights": weights},
        {"weights": benchmark},
        {"weights": dict(zip(inst["patients"], reversed(reference)))},
    ]
    w = lambda a: max(arithmetic(inst, v["weights"], a)[1] for v in vectors)
    robust = solver.robust_selection(session, inst, weights, optimum, vectors, 1e-6, 0, [31, 32])
    for selected in robust:
        assert w(selected["assignment"]) == pytest.approx(min(w(a) for a in allowed))
        assert solver.stress(inst, vectors, selected["assignment"]) == pytest.approx(
            w(selected["assignment"])
        )
        solver.verify(inst, weights, selected["assignment"], optimum, 1e-6)


def test_self_proxy_has_zero_regret_but_independent_samples_can_disagree(session):
    inst = instance()
    weights = dict.fromkeys(inst["patients"], 1)
    optimum = solver.nominal(session, inst, weights, "complexity-range", 1e-6)
    samples = [
        solver.random_selection(session, inst, weights, optimum, 1e-6, s)["assignment"]
        for s in range(12)
    ]
    assert len({tuple(a.items()) for a in samples}) > 1
    assert all(solver.actual(inst, weights, a)["range"] - optimum["g"] == 0 for a in samples)


def test_robust_optimizes_entire_set_and_positive_slack_preserves_count(session):
    inst = instance()
    weights = dict.fromkeys(inst["patients"], 1)
    alternative = dict(zip(inst["patients"], [1, 1, 3, 3]))
    optimum = solver.nominal(session, inst, weights, "lexicographic", 1e-6)
    vectors = [{"weights": weights}, {"weights": alternative}]
    selected = solver.robust_selection(session, inst, weights, optimum, vectors, 1e-6, 2, [9])[0]
    assert selected["robust_optimum"] == 0
    assert solver.actual(inst, weights, selected["assignment"])["count_range"] == 0


def test_incomplete_original_and_nonzero_gap_not_certified(monkeypatch):
    status = {
        "optimal": True,
        "has_incumbent": True,
        "objective": 2.0,
        "bound": 1.9,
        "status": "OPTIMAL",
        "gap": 0.05,
    }
    monkeypatch.setattr(solver, "stage_diagnostics", lambda *a: dict(status))
    result = solver.optimize(SimpleNamespace(optimize=lambda: None), SimpleNamespace(gp=None), 1e-6)
    assert not result["certified"]
    with pytest.raises(solver.IncompleteSolve):
        solver.require(result, "nominal")


def test_incomplete_envelope_keeps_bounds_without_exact_claim(session, monkeypatch):
    inst = instance()
    weights = dict.fromkeys(inst["patients"], 1)
    benchmark = dict(zip(inst["patients"], [1, 1, 3, 3]))
    optimum = solver.nominal(session, inst, weights, "complexity-range", 1e-6)
    bo = solver.nominal(session, inst, benchmark, "complexity-range", 1e-6)
    original = solver.optimize

    def interrupted(*args):
        result = original(*args)
        result.update(certified=False, optimal=False, status="TIME_LIMIT")
        return result

    monkeypatch.setattr(solver, "optimize", interrupted)
    result = solver.regret_envelope(session, inst, weights, benchmark, optimum, bo, 1e-6)
    assert not result["complete"]
    assert result["minimum_range_lower_bound"] is not None
    assert result["maximum_range_upper_bound"] is not None
