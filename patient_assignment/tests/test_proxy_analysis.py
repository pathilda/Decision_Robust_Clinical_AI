"""Independent small-instance oracles and saved-data/plot contract tests."""

import itertools
import json
import random
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
import yaml

from patient_assignment import experiment_solver as solver
from patient_assignment import proxy_analysis as workflow
from patient_assignment.optimization import SolverSession
from patient_assignment.proxy_cli import main, parser
from patient_assignment.proxy_inputs import agreement, membership, prepare, profiles
from patient_assignment.proxy_plots import build_figure, color, load_results, render
from patient_assignment.proxy_solver import ProxySettings, canonical, evaluate, select, stream_seed

DEMO = Path(__file__).resolve().parents[2] / "configs" / "proxy_demo"


def arguments(output, extra=(), demo=DEMO):
    return parser().parse_args(
        [
            "proxy-analysis",
            "--config",
            str(demo / "inputs.yaml"),
            "--models",
            "qwen_base",
            "qwen_fewshot",
            "--benchmark",
            "medgemma_base",
            "--chunks",
            str(demo / "manifest.json"),
            "--profiles",
            str(demo / "profiles.yaml"),
            "--n-samples",
            "3",
            "--chunk-seed",
            "99",
            "--output",
            str(output),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def session():
    # This suite deliberately fails (does not skip) when the required solver is unavailable.
    with SolverSession(ProxySettings()) as active:
        yield active


def tiny():
    return {
        "chunk_id": "01",
        "profile_id": "balanced",
        "patients": ["001", "002", "003", "004"],
        "clinicians": [
            {"id": c, "initial_count": 0, "initial_complexity": 0, "capacity": 4}
            for c in ("A", "B")
        ],
    }


def enumerate_optimal(instance, weights, mode):
    candidates = []
    for labels in itertools.product([c["id"] for c in instance["clinicians"]], repeat=len(weights)):
        chosen = dict(zip(instance["patients"], labels))
        counts, loads = [], []
        for c in instance["clinicians"]:
            patients = [p for p in chosen if chosen[p] == c["id"]]
            counts.append(c["initial_count"] + len(patients))
            loads.append(c["initial_complexity"] + sum(weights[p] for p in patients))
        if any(n > c["capacity"] for n, c in zip(counts, instance["clinicians"])):
            continue
        objective = (max(loads) - min(loads),)
        if mode == "lexicographic":
            objective = (max(counts) - min(counts), *objective)
        candidates.append((objective, chosen))
    best = min(o for o, a in candidates)
    return best, [a for o, a in candidates if o == best]


@pytest.mark.parametrize("mode", ["complexity-range", "lexicographic"])
@pytest.mark.parametrize("scores", [[3, 1, 1, 1], [1, 2, 3, 2], [0.1, 0.2, 0.3, 0.2]])
def test_exact_optima_and_random_objective_oracle(session, mode, scores):
    instance = tiny()
    weights = dict(zip(instance["patients"], scores))
    best, allowed = enumerate_optimal(instance, weights, mode)
    optimum = solver.nominal(session, instance, weights, mode, 1e-7)
    assert optimum["g"] == pytest.approx(best[-1])
    if mode == "lexicographic":
        assert optimum["d"] == best[0]
    for seed in (1, 7, 42, 99):
        result = select(session, instance, weights, optimum, 1e-7, seed)
        chosen = result["assignment"]
        assert chosen in allowed
        rng = random.Random(stream_seed(instance, seed))
        coefficients = {
            (p, c["id"]): rng.random() for p in sorted(weights) for c in instance["clinicians"]
        }
        objective = lambda a, coefficients=coefficients: sum(
            coefficients[p, a[p]] for p in sorted(weights)
        )
        assert objective(chosen) == pytest.approx(min(objective(a) for a in allowed))
        reordered = {
            **instance,
            "patients": list(reversed(instance["patients"])),
            "clinicians": list(reversed(instance["clinicians"])),
        }
        assert select(session, reordered, weights, optimum, 1e-7, seed)["assignment"] == chosen


def test_fixed_reference_patient_agreement_and_regret(session):
    instance = tiny()
    weights = dict.fromkeys(instance["patients"], 1)
    optimum = solver.nominal(session, instance, weights, "complexity-range", 1e-7)
    draws = [select(session, instance, weights, optimum, 1e-7, s)["assignment"] for s in range(12)]
    assert len({tuple(a.items()) for a in draws}) > 1
    reference = draws[0]
    values = [evaluate(instance, weights, optimum, a, reference) for a in draws]
    assert all(v["proxy_regret"] == 0 for v in values)
    assert any(v["exact_assignment_pct"] < 100 for v in values)
    changed = dict(reference)
    patient = instance["patients"][0]
    changed[patient] = "B" if changed[patient] == "A" else "A"
    assert evaluate(instance, weights, optimum, changed, reference)["exact_assignment_pct"] == 75
    swapped = {p: "B" if c == "A" else "A" for p, c in reference.items()}
    assert evaluate(instance, weights, optimum, swapped, reference)["exact_assignment_pct"] == 0
    proxy_weights = dict(zip(instance["patients"], [1, 1, 3, 3]))
    proxy_optimum = solver.nominal(session, instance, proxy_weights, "complexity-range", 1e-7)
    for a in draws:
        expected = (
            abs(
                sum(proxy_weights[p] for p in a if a[p] == "A")
                - sum(proxy_weights[p] for p in a if a[p] == "B")
            )
            - proxy_optimum["g"]
        )
        first = evaluate(instance, proxy_weights, proxy_optimum, a, reference)
        second = evaluate(instance, proxy_weights, proxy_optimum, a, swapped)
        assert first["proxy_regret"] == second["proxy_regret"] == expected


def test_qwk_hand_examples():
    assert agreement(["a", "b", "d"], ["a", "d", "d"], ["a", "b", "c", "d"])[1] == pytest.approx(
        2 / 3
    )
    assert agreement(["a", "b", "c"], ["a", "c", "c"], ["a", "b", "c"])[:2] == pytest.approx(
        (200 / 3, 0.8)
    )
    # Absent middle category stays at ordinal position 1 in the full order.
    assert agreement(["a", "c", "c"], ["a", "a", "c"], ["a", "b", "c"])[1] == pytest.approx(0.4)
    assert agreement(["a"], ["a"], ["a", "b", "c"])[1:] == (
        None,
        "undefined: zero expected weighted disagreement",
    )
    assert agreement(["a"], ["a"], ["a"])[1] is None


def test_inputs_and_cli_validation(tmp_path):
    prepared = prepare(arguments(tmp_path / "out"))
    assert prepared["resolved"]["chunks"]["group_1"][0] == "001"
    assert all(a["n_patients"] == 12 for a in prepared["accuracy"])
    for extra, match in [
        (["--models", "medgemma_base"], "must not"),
        (["--models", "qwen_base", "qwen_base"], "unique"),
        (["--selection-seeds", "1", "2"], "mutually exclusive"),
        (["--n-samples", "0"], ">= 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            prepare(arguments(tmp_path / "out", extra))
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({"chunks": {"a": ["001"], "b": ["001"]}}))
    with pytest.raises(ValueError, match="unique"):
        membership(path)
    config = tmp_path / "bad_profiles.yaml"
    raw = yaml.safe_load((DEMO / "profiles.yaml").read_text())
    raw["profiles"]["balanced"]["clinicians"][0]["initial_complexity"] = 31
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="scale-incompatible"):
        profiles(config, [1, 2, 3])


@pytest.mark.parametrize("issue", ["missing", "duplicate", "failed", "continuous"])
def test_bad_prediction_records_rejected(tmp_path, issue):
    demo = tmp_path / "inputs"
    shutil.copytree(DEMO, demo)
    table = pd.read_csv(demo / "qwen.csv", dtype=str)
    if issue == "missing":
        table = table.iloc[1:]
    elif issue == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif issue == "failed":
        table["success"] = "true"
        table.loc[0, "success"] = "false"
    else:
        table.loc[0, "treatment_category"] = "1.7"
    table.to_csv(demo / "qwen.csv", index=False)
    with pytest.raises(ValueError):
        prepare(arguments(tmp_path / "out", demo=demo))


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp("proxy") / "run"
    assert workflow.run(arguments(path)) == 0
    return path


@pytest.mark.parametrize("mode", ["Point", "Box"])
def test_figure_counts_groups_colors_and_offline_files(completed, tmp_path, mode):
    resolved, _report, frame = load_results(completed)
    assert frame.is_seed_sample.sum() == 2 * 2 * 3 * 3
    assert frame.is_chunk_evaluation.sum() == 2 * 2 * 3
    assert len(frame) == 48
    for faceted in (True, False):
        fig = build_figure(
            frame,
            resolved,
            resolved["models"],
            list(resolved["chunks"]),
            list(resolved["profiles"]),
            mode,
            1,
            "proxy_regret",
            faceted,
        )
        assert sum(len(t.y) for t in fig.data) == (36 if faceted else 12)
        assert {t.marker.color for t in fig.data if t.meta["model"] == "qwen_base"} == {
            color("qwen_base")
        }
        for trace in fig.data:
            assert all(c[3] == trace.meta["profile"] for c in trace.customdata)
            if mode == "Box":
                assert trace.quartilemethod == "linear"
        if faceted:
            assert fig.layout.yaxis.range == fig.layout.yaxis2.range
    render(completed, tmp_path / mode, mode, 3)
    files = list((tmp_path / mode / "figures").glob("*.html"))
    assert len(files) == 6
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert 'src="https://' not in text
        if path.stem.startswith("regret_vs_"):
            assert "plotly.js v" in text
            assert 'data-filter="profile"' in text
            assert 'data-filter="model_policy"' in text
            continue
        assert "plotly.js v" in text and "Starting Parameter" in text
        assert "hidden.has(t.meta.model)" in text and "plotly_legenddoubleclick" in text
        # Exercise the actual offline HTML controller in Node, including all facets.
        node = shutil.which("node")
        if node:
            tail = text[text.rfind("Plotly.newPlot(") :]
            start = tail.index(",") + 1
            data, _ = json.JSONDecoder().raw_decode(tail[start:].lstrip())
            script = text[text.rfind("<script>") + len("<script>") : text.rfind("</script>")]
            result = subprocess.run(
                [node, str(Path(__file__).with_name("proxy_visibility.cjs"))],
                input=json.dumps({"script": script, "data": data}),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, result.stderr


def test_reference_cache_reproducibility_resume_and_no_solver_plots(
    completed, tmp_path, monkeypatch
):
    original = pd.read_csv(completed / "results.csv")
    refs = pd.read_csv(completed / "reference_allocations.csv", dtype=str)
    assert refs.groupby(["chunk_id", "profile_id"]).solution_id.nunique().eq(1).all()
    monkeypatch.setattr(
        workflow, "SolverSession", lambda *a: pytest.fail("unexpected solver initialization")
    )
    assert (
        workflow.run(arguments(completed, ["--resume", "--models", "qwen_fewshot", "qwen_base"]))
        == 0
    )
    after = pd.read_csv(completed / "results.csv")
    keys = ["model_id", "chunk_id", "profile_id", "selection_seed"]
    assert (
        original.set_index(keys).solution_hash.to_dict()
        == after.set_index(keys).solution_hash.to_dict()
    )
    with pytest.raises(ValueError, match="Incompatible resume"):
        workflow.run(arguments(completed, ["--resume", "--benchmark-seed", "1002"]))
    # All matching artifacts reused with one seed alone and changed candidate order/count.
    single = arguments(
        tmp_path / "single",
        [
            "--n-samples",
            "1",
            "--sample-seed",
            "99",
            "--models",
            "qwen_base",
            "--cache-dir",
            str(completed / "cache"),
        ],
    )
    assert workflow.run(single) == 0
    one = pd.read_csv(single.output / "results.csv")
    expected = original.loc[original.model_id.eq("qwen_base") & original.selection_seed.eq(99)]
    assert (
        one.set_index(keys).solution_hash.to_dict()
        == expected.set_index(keys).solution_hash.to_dict()
    )
    assert set(one.reference_solution_id) == set(expected.reference_solution_id)
    # Import hook makes any attempt to initialize/import the solver API fail.
    monkeypatch.setitem(sys.modules, "gurobipy", None)
    assert (
        main(
            [
                "proxy-plots",
                "--results",
                str(completed),
                "--show_type",
                "Box",
                "--models",
                "qwen_base",
                "--starting-parameters",
                "balanced",
                "--chunk-ids",
                "group_1",
                "--output",
                str(tmp_path / "plots"),
            ]
        )
        == 0
    )


def test_missing_selection_data_rejected(completed, tmp_path):
    target = tmp_path / "broken"
    shutil.copytree(completed, target)
    (target / "results.csv").write_text("model_id\n")
    with pytest.raises(ValueError, match="corrupt"):
        load_results(target)


def test_infeasible_and_uncertified_instances_are_not_zero(tmp_path, monkeypatch):
    demo = tmp_path / "inputs"
    shutil.copytree(DEMO, demo)
    raw = yaml.safe_load((demo / "profiles.yaml").read_text())
    for c in raw["profiles"]["balanced"]["clinicians"]:
        c["capacity"] = c["initial_count"]
    (demo / "profiles.yaml").write_text(yaml.safe_dump(raw))
    args = arguments(tmp_path / "infeasible", ["--starting-parameters", "balanced"], demo)
    monkeypatch.setattr(
        workflow, "SolverSession", lambda *a: pytest.fail("infeasible must be preflighted")
    )
    assert workflow.run(args) == 2
    _, report, frame = load_results(args.output)
    assert frame.proxy_regret.isna().all() and report["infeasible_instances"] == 2
    assert "No certified observations" in next((args.output / "figures").glob("*.html")).read_text(
        encoding="utf-8"
    )


def test_time_limit_incumbents_rejected(session, monkeypatch):
    instance = tiny()
    weights = dict.fromkeys(instance["patients"], 1)
    optimum = solver.nominal(session, instance, weights, "complexity-range", 1e-7)
    original = solver.optimize

    def incomplete(*args):
        result = original(*args)
        result.update(certified=False, status="TIME_LIMIT", optimal=False)
        return result

    monkeypatch.setattr(solver, "optimize", incomplete)
    with pytest.raises(solver.IncompleteSolve, match="TIME_LIMIT"):
        select(session, instance, weights, optimum, 1e-7, 42)


def test_candidate_draws_independent_of_benchmark_and_reference_seed(session):
    instance = tiny()
    weights = dict.fromkeys(instance["patients"], 1)
    optimum = solver.nominal(session, instance, weights, "complexity-range", 1e-7)
    candidate = select(session, instance, weights, optimum, 1e-7, 42)["assignment"]
    regrets = []
    for seed in (1001, 1002):
        reference = select(session, instance, weights, optimum, 1e-7, seed, "benchmark")[
            "assignment"
        ]
        regrets.append(evaluate(instance, weights, optimum, candidate, reference)["proxy_regret"])
    assert regrets == [0, 0]
    assert canonical(instance)["patients"] == sorted(weights)


def test_workflow_row_order_benchmark_and_reference_invariance(session, tmp_path, monkeypatch):
    from patient_assignment import proxy_plots

    demo = tmp_path / "inputs"
    shutil.copytree(DEMO, demo)
    monkeypatch.setattr(workflow, "SolverSession", lambda settings: nullcontext(session))
    monkeypatch.setattr(proxy_plots, "render", lambda *a, **kw: None)

    def execute(name, extra=()):
        args = arguments(
            tmp_path / name,
            [
                "--starting-parameters",
                "balanced",
                "--chunk-ids",
                "group_1",
                "--cache-dir",
                str(tmp_path / "cache"),
                *extra,
            ],
            demo,
        )
        assert workflow.run(args) == 0
        frame = pd.read_csv(args.output / "results.csv")
        return frame.set_index(["model_id", "chunk_id", "profile_id", "selection_seed"])

    first = execute("first")
    for name in ("qwen.csv", "benchmark.csv", "fewshot.csv"):
        table = pd.read_csv(demo / name, dtype=str)
        table.iloc[::-1].to_csv(demo / name, index=False)
    reordered = execute("reordered", ["--models", "qwen_fewshot", "qwen_base"])
    assert first.solution_hash.to_dict() == reordered.solution_hash.to_dict()
    changed_reference = execute("reference", ["--benchmark-seed", "1002"])
    assert first.solution_hash.to_dict() == changed_reference.solution_hash.to_dict()
    assert first.proxy_regret.to_dict() == changed_reference.proxy_regret.to_dict()
    changed_benchmark = execute(
        "benchmark", ["--models", "qwen_base", "--benchmark", "qwen_fewshot"]
    )
    assert (
        first.loc["qwen_base"].solution_hash.to_dict()
        == changed_benchmark.loc["qwen_base"].solution_hash.to_dict()
    )
