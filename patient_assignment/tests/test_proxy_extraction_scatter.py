import pandas as pd
import pytest

from proxy_extraction_scatter import build_scatter, scatter_page, write_scatters


def example():
    rows = []
    for model in ["a", "b"]:
        for policy in ["deterministic", "stochastic"]:
            for profile in ["balanced", "unbalanced"]:
                for seed in [42, 43]:
                    rows.append(
                        {
                            "model_id": model,
                            "policy": policy,
                            "policy_configuration_id": policy,
                            "profile_id": profile,
                            "chunk_id": "01",
                            "selection_seed": seed,
                            "solver_status": "OPTIMAL",
                            "proxy_regret": 2,
                            "solution_hash": "repeated",
                        }
                    )
    scores = pd.DataFrame(
        [
            {"model_id": "a", "qwk": 0.8, "exact_accuracy_pct": 90},
            {"model_id": "b", "qwk": 0.4, "exact_accuracy_pct": 60},
        ]
    )
    resolved = {
        "profiles": {
            p: [{"id": "A", "initial_count": n}] for p, n in [("balanced", 2), ("unbalanced", 7)]
        }
    }
    return pd.DataFrame(rows), scores, resolved


@pytest.mark.parametrize("metric", ["qwk", "exact_accuracy_pct"])
def test_every_allocation_preserved_with_correct_model_score(metric):
    frame, scores, resolved = example()
    fig = build_scatter(frame, scores, resolved, metric)
    assert sum(len(t.x) for t in fig.data) == len(frame)
    assert sum(bool(t.showlegend) for t in fig.data) == 4
    for trace in fig.data:
        model = trace.meta["model_policy"].split(" / ")[0]
        expected = scores.set_index("model_id").loc[model, metric]
        assert set(trace.x) == {expected}
        assert set(trace.y) == {2}
        assert trace.meta["shown"] == 2
        assert trace.customdata[0][8] in {"A: 2", "A: 7"}
    page = scatter_page(fig, metric)
    assert 'data-filter="profile"' in page
    assert 'data-filter="model_policy"' in page
    assert "hidden.has(t.legendgroup)" in page
    assert "const showlegend" in page
    assert "<script src=" not in page


def test_missing_scores_and_failed_rows_are_counted_not_plotted():
    frame, scores, resolved = example()
    scores.loc[scores.model_id == "b", "qwk"] = float("nan")
    frame.loc[0, "solver_status"] = "FAILED"
    fig = build_scatter(frame, scores, resolved, "qwk")
    assert sum(len(t.x) for t in fig.data) == 7
    assert sum(t.meta["missing"] for t in fig.data) == 8
    assert sum(t.meta["failed"] for t in fig.data) == 1
    assert sum(t.meta["total"] for t in fig.data) == 16


def test_legacy_defaults_and_output_links(tmp_path):
    frame, scores, resolved = example()
    frame = frame[frame.policy == "deterministic"].drop(
        columns=["policy", "policy_configuration_id"]
    )
    links = write_scatters(frame, scores, resolved, tmp_path)
    assert len(links) == 2
    assert len(list((tmp_path / "figures").glob("*.html"))) == 2
    fig = build_scatter(frame, scores, resolved, "qwk")
    assert all(t.meta["model_policy"].endswith("/ deterministic") for t in fig.data)


def test_duplicate_extraction_rows_rejected():
    frame, scores, resolved = example()
    with pytest.raises(pd.errors.MergeError):
        build_scatter(frame, pd.concat([scores, scores]), resolved, "qwk")
