"""Independent enumeration oracles and synthetic uncertainty workflow contracts.

Pure tests need pytest/pandas/PyYAML. Solver tests skip explicitly without Gurobi/license.
"""

import itertools
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from patient_assignment import experiment_solver as s
from patient_assignment import proxy_policy_solver as ps
from patient_assignment.optimization import SolverDependencyError, SolverSession
from patient_assignment.proxy_cli import main, parser
from patient_assignment.proxy_inputs import prepare
from patient_assignment.proxy_policy_analysis import PolicyCache, problem_for, run
from patient_assignment.proxy_solver import ProxySettings
from patient_assignment.proxy_uncertainty import configuration, load_probabilities

ROOT = Path(__file__).resolve().parents[2]


def instance(n=4, clinicians=2):
    return {
        "chunk_id": "synthetic",
        "profile_id": "balanced",
        "patients": [str(i) for i in range(n)],
        "clinicians": [
            {"id": chr(65 + j), "initial_count": 2, "initial_complexity": 2 + j, "capacity": n + 2}
            for j in range(clinicians)
        ],
    }


def assignments(i):
    for values in itertools.product([c["id"] for c in i["clinicians"]], repeat=len(i["patients"])):
        chosen = dict(zip(i["patients"], values))
        if all(
            c["initial_count"] + values.count(c["id"]) <= c["capacity"] for c in i["clinicians"]
        ):
            yield chosen


def range_oracle(i, weights, chosen):
    loads = [
        c["initial_complexity"] + sum(weights[p] for p in i["patients"] if chosen[p] == c["id"])
        for c in i["clinicians"]
    ]
    return max(loads) - min(loads)


def robust_oracle(i, nominal, supports, budget, chosen):
    return max(
        range_oracle(i, dict(zip(i["patients"], values)), chosen)
        for values in itertools.product(*(supports[p] for p in i["patients"]))
        if sum(v != nominal[p] for p, v in zip(i["patients"], values)) <= budget
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("clinicians", [2, 3])
@pytest.mark.parametrize("weights", [(1, 2, 3), (0.2, 1.7, 4.3)])
def test_exact_separation_against_independent_cartesian_oracle(n, clinicians, weights):
    i = instance(n, clinicians)
    nominal = {p: weights[int(p) % 3] for p in i["patients"]}
    supports = {
        p: list(weights) if int(p) % 2 else sorted({nominal[p], weights[0], weights[-1]})
        for p in i["patients"]
    }
    chosen = dict(zip(i["patients"], [chr(65 + k % clinicians) for k in range(n)]))
    for budget in range(n + 1):
        value, witness = ps.separate(i, nominal, supports, budget, chosen)
        assert value == pytest.approx(robust_oracle(i, nominal, supports, budget, chosen))
        assert range_oracle(i, witness, chosen) == pytest.approx(value)
        assert sum(witness[p] != nominal[p] for p in nominal) <= budget


def test_support_nominal_inclusion_and_threshold_nesting():
    p = {"a": [0.1, 0.2, 0.7], "b": [0.0, 1.0, 0.0]}
    nominal = {"a": 1, "b": 3}
    previous = ps.support(nominal, p, [1, 2, 3], 0)
    for threshold in (0.05, 0.2, 0.5, 1.0):
        current = ps.support(nominal, p, [1, 2, 3], threshold)
        for patient in p:
            assert nominal[patient] in current[patient]
            assert set(current[patient]) <= set(previous[patient])
        previous = current


def test_scenario_seed_roles_ordering_and_nonanticipation():
    probabilities = {"b": [0.2, 0.5, 0.3], "a": [0.1, 0.3, 0.6]}
    args = (probabilities, [1, 2, 3], "Qwen", "chunk", 42, 100)
    first = ps.scenarios(*args)
    assert first == ps.scenarios(*args)
    assert first == ps.scenarios(dict(reversed(list(probabilities.items()))), *args[1:])
    assert sum(first["multiplicities"]) == 100
    assert sum(n / 100 for n in first["multiplicities"]) == pytest.approx(1)
    assert (
        first["fingerprint"]
        != ps.scenarios(probabilities, [1, 2, 3], "Qwen", "chunk", 43, 100)["fingerprint"]
    )
    assert first["fingerprint"] != ps.scenarios(*args, role="holdout")["fingerprint"]
    assert all(set(v) == set(probabilities) for v in first["vectors"])
    # Neither profile nor selection seed is a generator argument.
    i = instance(2)
    probs = {p: [0.2, 0.5, 0.3] for p in i["patients"]}
    cell = {"policy": "stochastic", "scenario_count": 20, "scenario_seed": 42, "temperature": 1.0}
    a, _ = problem_for(cell, i, "M", dict.fromkeys(i["patients"], 2), probs, [1, 2, 3])
    i["profile_id"] = "unbalanced"
    b, _ = problem_for(cell, i, "M", dict.fromkeys(i["patients"], 2), probs, [1, 2, 3])
    assert a == b


def test_expected_range_is_not_range_of_expected_severity():
    i = instance(2)
    for c in i["clinicians"]:
        c["initial_complexity"] = 0
    chosen = {"0": "A", "1": "B"}
    vectors = [{"0": a, "1": b} for a, b in itertools.product([1, 3], repeat=2)]
    mean_range = sum(range_oracle(i, v, chosen) for v in vectors) / 4
    assert mean_range == 1
    assert range_oracle(i, {"0": 2, "1": 2}, chosen) == 0


def test_holdout_cvar_fractional_boundary():
    i = instance(1)
    for c in i["clinicians"]:
        c["initial_complexity"] = 0
    data = {
        "vectors": [{"0": v} for v in [1, 2, 3]],
        "multiplicities": [1, 1, 1],
        "count": 3,
        "fingerprint": "test",
        "seed": 99,
    }
    metrics = ps.holdout_metrics(i, data, {"0": "A"}, 0.5)
    assert metrics["holdout_mean"] == 2
    assert metrics["holdout_cvar"] == pytest.approx(8 / 3)
    assert metrics["holdout_std"] == pytest.approx(math.sqrt(2 / 3))


def args_for(tmp_path, extra=()):
    demo = ROOT / "configs" / "proxy_uncertainty_demo"
    return parser().parse_args(
        [
            "proxy-analysis",
            "--config",
            str(demo / "inputs.yaml"),
            "--models",
            "Qwen",
            "MedGemma",
            "OpenDecider",
            "--benchmark",
            "proxy",
            "--chunks",
            str(demo / "chunks.csv"),
            "--profiles",
            str(demo / "profiles.yaml"),
            "--policies",
            "deterministic",
            "stochastic",
            "robust",
            "--uncertainty-config",
            str(demo / "pilot.yaml"),
            "--n-samples",
            "3",
            "--chunk-seed",
            "42",
            "--output",
            str(tmp_path / "results"),
            *extra,
        ]
    )


def test_grid_dimensions_and_strict_validation(tmp_path):
    _config, grid = configuration(None, ["deterministic", "stochastic", "robust"])
    assert len(grid) == 1 + 4 * 3 + 5 * 2
    assert len({c["policy_configuration_id"] for c in grid}) == len(grid)
    assert grid == configuration(None, ["deterministic", "stochastic", "robust"])[1]
    path = tmp_path / "bad.yaml"
    for bad in [
        {"schema_version": 2},
        {"schema_version": 1, "unknown": True},
        {"schema_version": 1, "robust": {"gamma_rounding": "floor"}},
        {"schema_version": 1, "stochastic": {"scenario_counts": [0]}},
        {"schema_version": 1, "evaluation": {"cvar_alpha": 1}},
        {"schema_version": 1, "probabilities": {"temperature_values": [0]}},
    ]:
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            configuration(path, ["stochastic"])


def test_probability_inputs_and_deterministic_backward_compatibility(tmp_path):
    args = args_for(tmp_path)
    prepared = prepare(args)
    config, _ = configuration(args.uncertainty_config, args.policies)
    data, diagnostics, sources = load_probabilities(args, prepared, config)
    assert set(data) == {"Qwen", "MedGemma", "OpenDecider"}
    assert all(len(d) == 8 for d in data.values())
    assert any(d["point_argmax_disagreements"] for d in diagnostics)
    assert all("sha256" in s for s in sources.values())
    # The existing categorical-only synthetic registry still loads without probabilities.
    legacy = ROOT / "configs" / "proxy_demo"
    args.config, args.chunks, args.profiles = (
        legacy / "inputs.yaml",
        legacy / "manifest.json",
        legacy / "profiles.yaml",
    )
    args.models, args.benchmark = ["qwen_base"], "medgemma_base"
    assert prepare(args)["loaded"]["qwen_base"]


@pytest.mark.parametrize(
    "bad", ["duplicate", "missing", "nan", "negative", "over_one", "sum", "columns"]
)
def test_probability_failures(tmp_path, bad):
    args = args_for(tmp_path)
    registry = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    for entry in registry["inputs"]:
        entry["path"] = str((args.config.parent / entry["path"]).resolve())
        if "probability_path" in entry:
            entry["probability_path"] = str(
                (args.config.parent / entry["probability_path"]).resolve()
            )
    entry = registry["inputs"][0]
    frame = pd.read_csv(entry["path"], dtype=str)
    if bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif bad == "missing":
        frame = frame.iloc[1:]
    elif bad == "columns":
        frame = frame.drop(columns=["probability_easy"])
    else:
        frame.loc[0, "probability_easy"] = {
            "nan": "NaN",
            "negative": "-.1",
            "over_one": "1.1",
            "sum": ".95",
        }[bad]
    source = tmp_path / "probabilities.csv"
    frame.to_csv(source, index=False)
    entry["probability_path"] = str(source)
    args.config = tmp_path / "registry.yaml"
    args.config.write_text(yaml.safe_dump(registry))
    prepared = prepare(args)
    config, _ = configuration(args.uncertainty_config, args.policies)
    with pytest.raises(ValueError):
        load_probabilities(args, prepared, config)


def test_explicit_renormalization_and_disagreement_error(tmp_path):
    args = args_for(tmp_path)
    prepared = prepare(args)
    config, _ = configuration(args.uncertainty_config, args.policies)
    config["probabilities"]["point_prediction_consistency"] = "error"
    with pytest.raises(ValueError, match="disagrees"):
        load_probabilities(args, prepared, config)


def test_cache_identity_and_interruption(tmp_path):
    cache = PolicyCache(tmp_path / "cache")

    def fail():
        raise RuntimeError("forced interruption")

    with pytest.raises(RuntimeError):
        cache.get({"confidence": 0.7}, fail)
    assert not list(cache.path.glob("*.json"))
    assert cache.get({"confidence": 0.7}, lambda: {"certified": True}) == {"certified": True}
    assert cache.get({"confidence": 0.7}, fail) == {"certified": True}
    assert cache.get({"confidence": 0.8}, lambda: {"different": True}) == {"different": True}
    assert cache.hits == 1


def test_dry_run_without_solver(tmp_path, monkeypatch, capsys):
    import patient_assignment.proxy_policy_analysis as analysis

    monkeypatch.setattr(analysis, "SolverSession", lambda *_: pytest.fail("dry-run started solver"))
    assert run(args_for(tmp_path, ["--dry-run"])) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["planned"]["policy_cells"] == 3 * 2 * 3 * 4
    assert report["planned"]["selection_attempts"] == 3 * report["planned"]["policy_cells"]
    assert not (tmp_path / "results").exists()


@pytest.fixture(scope="module")
def session():
    pytest.importorskip("gurobipy", reason="Gurobi Python API is unavailable")
    try:
        with SolverSession(ProxySettings(time_limit=30)) as active:
            yield active
    except SolverDependencyError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("mode", ["complexity-range", "lexicographic"])
@pytest.mark.parametrize("policy", ["stochastic", "robust"])
def test_policy_master_and_random_selections_against_enumeration(session, mode, policy):
    i = instance()
    nominal = dict(zip(i["patients"], [1, 2, 3, 1]))
    probs = {p: [0.2, 0.5, 0.3] for p in i["patients"]}
    problem = {"policy": policy}
    if policy == "stochastic":
        problem["scenarios"] = ps.scenarios(probs, [1, 2, 3], "M", "tiny", 7, 20)

        def loss(a):
            d = problem["scenarios"]
            return (
                sum(range_oracle(i, v, a) * n for v, n in zip(d["vectors"], d["multiplicities"]))
                / d["count"]
            )
    else:
        problem.update(supports=ps.support(nominal, probs, [1, 2, 3], 0.1), budget=2)

        def loss(a):
            return robust_oracle(i, nominal, problem["supports"], problem["budget"], a)

    candidates = list(assignments(i))
    if mode == "lexicographic":
        count_opt = min(s.actual(i, nominal, a)["count_range"] for a in candidates)
        candidates = [a for a in candidates if s.actual(i, nominal, a)["count_range"] == count_opt]
    best = min(loss(a) for a in candidates)
    optimum = ps.solve(session, i, nominal, problem, mode, 1e-7)
    assert optimum["g"] == pytest.approx(best)
    for seed in [1, 7, 42]:
        selected = ps.solve(session, i, nominal, problem, mode, 1e-7, optimum, seed)
        assert selected["assignment"] in candidates
        assert loss(selected["assignment"]) == pytest.approx(best)
        assert len(selected["assignment"]) == len(nominal)  # one shared, nonanticipative assignment


@pytest.mark.parametrize("mode", ["complexity-range", "lexicographic"])
def test_gamma_zero_monotonicity_and_single_scenario(session, mode):
    i = instance(3)
    nominal = {"0": 1, "1": 2, "2": 3}
    baseline = s.nominal(session, i, nominal, mode, 1e-7)
    problem = {"policy": "robust", "supports": {key: [1, 2, 3] for key in nominal}, "budget": 0}
    previous = baseline["g"]
    for budget in range(4):
        problem["budget"] = budget
        optimum = ps.solve(session, i, nominal, problem, mode, 1e-7)
        if budget == 0:
            assert optimum["g"] == pytest.approx(baseline["g"])
        assert optimum["g"] >= previous - 1e-7
        previous = optimum["g"]
    problem = {
        "policy": "stochastic",
        "scenarios": {"vectors": [nominal], "multiplicities": [1], "count": 1},
    }
    assert ps.solve(session, i, nominal, problem, mode, 1e-7)["g"] == pytest.approx(baseline["g"])


@pytest.mark.parametrize("policy", ["stochastic", "robust"])
def test_multiple_optima_can_have_different_proxy_loss(session, policy):
    i = instance(4)
    for c in i["clinicians"]:
        c["initial_complexity"] = 0
    nominal = dict.fromkeys(i["patients"], 1)
    proxy = dict(zip(i["patients"], [1, 1, 3, 3]))
    problem = {
        "policy": policy,
        "supports": {p: [1] for p in nominal},
        "budget": 4,
        "scenarios": {"vectors": [nominal], "multiplicities": [1], "count": 1},
    }
    optimum = ps.solve(session, i, nominal, problem, "lexicographic", 1e-7)
    selected = [
        ps.solve(session, i, nominal, problem, "lexicographic", 1e-7, optimum, seed)
        for seed in range(20)
    ]
    assert all(ps.objective(i, nominal, problem, r["assignment"]) == optimum["g"] for r in selected)
    assert len({range_oracle(i, proxy, r["assignment"]) for r in selected}) > 1
    assert len({json.dumps(r["assignment"], sort_keys=True) for r in selected}) < len(selected)


def test_synthetic_end_to_end_resume_and_offline_plots(session, tmp_path, monkeypatch):
    import patient_assignment.proxy_policy_analysis as analysis

    args = args_for(tmp_path, ["--starting-parameters", "balanced", "--chunk-ids", "01"])
    assert run(args) == 0
    frame = pd.read_csv(args.output / "results.csv")
    assert len(frame) == 3 * 4 * 3
    assert frame.certified.all()
    assert (frame.proxy_regret >= -1e-7).all()
    assert frame.reference_solution_id.nunique() == 1
    assert frame.proxy_optimum.nunique() == 1
    for _, group in frame.groupby("model_id"):
        assert group.holdout_fingerprint.nunique() == 1
    saved = (args.output / "results.csv").read_bytes()
    monkeypatch.setattr(analysis, "SolverSession", lambda *_: pytest.fail("resume used solver"))
    args.resume = True
    assert run(args) == 0
    assert (args.output / "results.csv").read_bytes() == saved
    assert (
        main(
            [
                "proxy-plots",
                "--results",
                str(args.output),
                "--output",
                str(tmp_path / "plots"),
                "--show_type",
                "Box",
                "--policies",
                "stochastic",
                "robust",
            ]
        )
        == 0
    )
    page = (tmp_path / "plots" / "figures" / "regret_by_chunk_Box.html").read_text(encoding="utf-8")
    assert "plotly.js" in page
    assert "starting-parameter" in page
    assert "<script src=" not in page
    args.solver_seed += 1
    with pytest.raises(ValueError, match="Incompatible resume"):
        run(args)


def test_renormalization_aliases_and_probability_hash_changes(tmp_path):
    args = args_for(tmp_path)
    args.models = ["Qwen"]
    registry = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for entry in registry["inputs"]:
        entry["path"] = str((args.config.parent / entry["path"]).resolve())
    entry = registry["inputs"][0]
    source = tmp_path / "probabilities.csv"
    frame = pd.read_csv(entry["path"], dtype=str)
    frame.loc[0, "probability_easy"] = "0.70001"
    frame.to_csv(source, index=False)
    entry["probability_path"] = str(source)
    entry["probability_columns"] = {
        "Easy to treatment": "probability_easy",
        "Moderate to treatment": "probability_moderate",
        "severe to treatment": "probability_severe",
    }
    args.config = tmp_path / "registry.yaml"
    args.config.write_text(yaml.safe_dump(registry), encoding="utf-8")
    prepared = prepare(args)
    config, _ = configuration(args.uncertainty_config, args.policies)
    with pytest.raises(ValueError, match="renormalize"):
        load_probabilities(args, prepared, config)
    config["probabilities"]["renormalize"] = True
    data, diagnostics, sources = load_probabilities(args, prepared, config)
    assert diagnostics[0]["renormalized_rows"] == 1
    assert sum(data["Qwen"]["001"]) == pytest.approx(1)
    old_hash = sources["Qwen"]["normalized_sha256"]
    frame.loc[0, "probability_easy"] = "0.6"
    frame.loc[0, "probability_moderate"] = "0.3"
    frame.to_csv(source, index=False)
    assert load_probabilities(args, prepared, config)[2]["Qwen"]["normalized_sha256"] != old_hash


def test_failure_rows_and_empty_success_plots(tmp_path, monkeypatch):
    import patient_assignment.proxy_policy_analysis as analysis

    def unavailable():
        raise RuntimeError("Synthetic solver unavailability")

    monkeypatch.setattr(analysis, "SolverSession", lambda *_: unavailable())
    args = args_for(tmp_path, ["--starting-parameters", "balanced", "--chunk-ids", "01"])
    assert run(args) == 2
    frame = pd.read_csv(args.output / "results.csv")
    summary = pd.read_csv(args.output / "policy_summary.csv")
    assert len(frame) == 36
    assert not frame.certified.any()
    assert {
        "selected_policy_objective",
        "expected_scenario_objective",
        "robust_worst_case_objective",
        "holdout_seed",
    } <= set(frame.columns)
    assert "policy" in pd.read_csv(args.output / "allocations.csv").columns
    assert "holdout_cvar" in pd.read_csv(args.output / "holdout_metrics.csv").columns
    assert len(pd.read_csv(args.output / "run_failures.csv")) == 36
    assert summary.n_total.sum() == 36
    assert summary.n_success.sum() == 0
    assert (args.output / "index.html").is_file()


def test_forced_interruption_resumes_current_cells_only(session, tmp_path, monkeypatch):
    import patient_assignment.proxy_policy_analysis as analysis

    args = args_for(tmp_path, ["--starting-parameters", "balanced", "--chunk-ids", "01"])
    args.models = ["Qwen"]
    original = analysis.PolicyCache.get
    calls = 0

    def interrupted(self, identity, compute):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt("synthetic interruption")
        return original(self, identity, compute)

    monkeypatch.setattr(analysis.PolicyCache, "get", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(args)
    assert len(list((args.output / "cache").glob("*.json"))) == 5
    monkeypatch.setattr(analysis.PolicyCache, "get", original)
    args.resume = True
    assert run(args) == 0
    frame = pd.read_csv(args.output / "results.csv")
    assert len(frame) == 12
    assert not frame.duplicated(["model_id", "policy_configuration_id", "selection_seed"]).any()
    report = json.loads((args.output / "validation_report.json").read_text(encoding="utf-8"))
    assert report["cache_hits"] >= 5
    assert frame.run_fingerprint.nunique() == 1


def test_integer_budget_and_nonuniform_weights(session):
    i = instance(3, 3)
    nominal = {"0": 0.2, "1": 1.7, "2": 4.3}
    probabilities = {key: [0.2, 0.5, 0.3] for key in nominal}
    cell = {"policy": "robust", "gamma_count": 1, "probability_threshold": 0.1}
    problem, metadata = problem_for(cell, i, "M", nominal, probabilities, [0.2, 1.7, 4.3])
    assert metadata["gamma_effective"] == 1
    optimum = ps.solve(session, i, nominal, problem, "complexity-range", 1e-7)
    expected = min(
        robust_oracle(i, nominal, problem["supports"], 1, chosen) for chosen in assignments(i)
    )
    assert optimum["g"] == pytest.approx(expected)


@pytest.mark.parametrize("policy,expected", [("stochastic", 1.5), ("robust", 4.0)])
def test_uncertainty_policy_is_not_restricted_to_nominal_optima(session, policy, expected):
    i = instance(4)
    for c in i["clinicians"]:
        c["initial_complexity"] = 0
    nominal = dict(zip(i["patients"], [1, 1, 1, 3]))
    vectors = [dict(zip(i["patients"], [*v, 3])) for v in itertools.product([1, 3], repeat=3)]
    problem = {
        "policy": policy,
        "supports": {p: [1, 3] if p != "3" else [3] for p in nominal},
        "budget": 3,
        "scenarios": {"vectors": vectors, "multiplicities": [1] * 8, "count": 8},
    }
    baseline = s.nominal(session, i, nominal, "complexity-range", 1e-7)
    optimum = ps.solve(session, i, nominal, problem, "complexity-range", 1e-7)
    assert baseline["g"] == 0
    assert optimum["g"] == pytest.approx(expected)
    assert ps.objective(i, nominal, problem, baseline["assignment"]) > optimum["g"]
    assert range_oracle(i, nominal, optimum["assignment"]) > baseline["g"]


def test_solver_api_error_is_reportable_and_not_cached(tmp_path):
    gp = pytest.importorskip("gurobipy")
    cache = PolicyCache(tmp_path / "cache")

    def too_large():
        raise gp.GurobiError(10010, "synthetic size-limited license error")

    with pytest.raises(RuntimeError, match="10010"):
        cache.get({"policy": "robust"}, too_large)
    assert not list(cache.path.glob("*.json"))


def test_fractional_budget_ceil_uses_decimal_not_binary_rounding():
    i = instance(100)
    nominal = dict.fromkeys(i["patients"], 1)
    probabilities = {p: [0.5, 0.3, 0.2] for p in nominal}
    cell = {"policy": "robust", "gamma_fraction": 0.14, "probability_threshold": 0.05}
    problem, metadata = problem_for(cell, i, "M", nominal, probabilities, [1, 2, 3])
    # Binary float .14 * 100 is 14.000000000000002; its ceil would wrongly allow 15.
    assert problem["budget"] == metadata["gamma_effective"] == 14
