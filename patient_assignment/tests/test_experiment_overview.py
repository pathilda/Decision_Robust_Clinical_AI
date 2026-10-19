"""Saved-result overview aggregation and solver-free plotting."""

import json

import pandas as pd
import pytest

from patient_assignment import cli, optimization
from patient_assignment.experiment_plots import aggregate_overview


def test_overview_weights_chunks_not_selection_repeats_or_patients():
    base = {
        "model_id": "model",
        "objective": "complexity-range",
        "profile_id": "balanced",
        "strategy": "random",
        "oracle": False,
        "nominal_slack": 0,
        "certified": True,
        "disagreement": 0.2,
        "benchmark_range": 8,
        "count_range": 1,
    }
    rows = [{**base, "chunk_id": "c1", "patient_count": 20, "regret": 0}] * 40
    rows += [{**base, "chunk_id": "c2", "patient_count": 10, "regret": 10}]
    rows += [{**base, "chunk_id": "c2", "patient_count": 10, "regret": 1000, "certified": False}]
    chunks, means = aggregate_overview(pd.DataFrame(rows))
    assert len(chunks) == 2 and len(means) == 1
    assert means.regret.iloc[0] == 5  # (mean(c1) + mean(c2)) / 2
    assert means.chunks.iloc[0] == 2 and means.draws.iloc[0] == 41
    assert means.min_chunk_size.iloc[0] == 10 and means.max_chunk_size.iloc[0] == 20
    assert means.disagreement.iloc[0] == pytest.approx(0.2)


def make_saved_results(path):
    path.mkdir()
    profiles = {
        "balanced": [15, 15, 15],
        "unbalanced": [12.5, 15, 17.5],
        "highly_unbalanced": [10, 15, 20],
    }
    config = {
        "run_fingerprint": "synthetic-overview-test",
        "benchmark": {"id": "model_b", "kind": "proxy"},
        "scoring": {"scale": "additive complexity (1-2)"},
        "profiles": {
            pid: {"clinicians": [{"id": c, "initial_complexity": w} for c, w in zip("ABC", loads)]}
            for pid, loads in profiles.items()
        },
    }
    (path / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = []
    for m, mid in enumerate(["model_a", "model_b", "model_c", "model_d", "model_e"]):
        for mode in ("complexity-range", "lexicographic"):
            for i, profile in enumerate(profiles):
                for strategy in ("random", "matched"):
                    for chunk in ("c1", "c2", "c3"):
                        rows.append(
                            {
                                "model_id": mid,
                                "objective": mode,
                                "profile_id": profile,
                                "strategy": strategy,
                                "chunk_id": chunk,
                                "sample": 0,
                                "certified": True,
                                "oracle": False,
                                "nominal_slack": 0,
                                "patient_count": 20,
                                "regret": m * 0.2 + i * 0.5,
                                "disagreement": 0.15 + m * 0.1 + i * 0.05,
                                "benchmark_range": 2 + i + m,
                                "count_range": 1 if mode == "lexicographic" else i + 1,
                            }
                        )
    pd.DataFrame(rows).to_csv(path / "solutions.csv", index=False)
    pd.DataFrame(columns=["model_id", "objective", "chunk_id", "profile_id", "complete"]).to_csv(
        path / "regret_bounds.csv", index=False
    )


def test_overview_outputs_two_figures_per_strategy_without_solver(tmp_path, monkeypatch):
    root, output = tmp_path / "results", tmp_path / "plots"
    make_saved_results(root)
    monkeypatch.setattr(optimization, "gurobi", lambda: pytest.fail("Plotting initialized solver"))
    assert (
        cli.main(
            [
                "assignment-plots",
                "--results",
                str(root),
                "--output",
                str(output),
                "--plots",
                "overview",
                "--selection",
                "random",
                "--format",
                "png",
                "pdf",
                "--max-instances-per-figure",
                "1",
            ]
        )
        == 0
    )
    assert len(list(output.glob("*.png"))) == 2
    assert len(list(output.glob("*.pdf"))) == 2
    means = pd.read_csv(output / "overview_means.csv")
    assert len(means) == 30 and means.chunks.eq(3).all()
    assert set(means.strategy) == {"random"}
    manifest = json.loads((output / "plot_manifest.json").read_text())
    assert len(manifest["files"]) == 4
    assert "equal chunk weights" in manifest["statistics"]
    # Source results remain untouched.
    assert len(pd.read_csv(root / "solutions.csv")) == 180


def test_overview_keeps_objectives_and_matched_diagnostics_separate():
    rows = []
    for mode, value in (("complexity-range", 1), ("lexicographic", 3)):
        for strategy, offset in (("random", 0), ("matched", 10)):
            for chunk in ("c1", "c2"):
                rows.append(
                    {
                        "model_id": "m",
                        "objective": mode,
                        "profile_id": "balanced",
                        "strategy": strategy,
                        "chunk_id": chunk,
                        "certified": True,
                        "oracle": False,
                        "nominal_slack": 0,
                        "patient_count": 20,
                        "regret": value + offset,
                        "disagreement": 0.2,
                        "benchmark_range": 3,
                        "count_range": 1,
                    }
                )
    _, means = aggregate_overview(pd.DataFrame(rows))
    assert len(means) == 4
    assert sorted(means.regret) == [1, 3, 11, 13]
