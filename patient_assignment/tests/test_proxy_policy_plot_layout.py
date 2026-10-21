import pandas as pd
import pytest

from patient_assignment.proxy_policy_plots import chart, standalone


def data():
    return pd.DataFrame(
        [
            {
                "model_id": model,
                "policy": policy,
                "policy_configuration_id": policy + "-hash",
                "chunk_id": chunk,
                "profile_id": "balanced",
                "certified": True,
                "solution_hash": chunk,
                "exact_assignment_pct": 50,
                "selection_seed": 42,
            }
            for model in ["Qwen", "OpenDecider"]
            for policy in ["deterministic", "stochastic", "robust"]
            for chunk in ["01", "02", "03"]
        ]
    )


@pytest.mark.parametrize("mode", ["Point", "Box"])
@pytest.mark.parametrize("faceted", [True, False])
def test_one_legend_entry_per_model_policy_preserves_observations(mode, faceted):
    frame = data()
    fig = chart(frame, "exact_assignment_pct", "method", mode, 3, faceted)
    assert len(fig.data) == 18
    assert sum(len(t.y) for t in fig.data) == len(frame)
    assert sum(bool(t.showlegend) for t in fig.data) == 6
    assert len({t.legendgroup for t in fig.data}) == 6
    assert all("-hash" not in t.name for t in fig.data)
    assert fig.layout.legend.orientation == "v"
    assert fig.layout.height >= 650
    assert tuple(fig.layout.yaxis.range) == (0, 100)


def test_filter_updates_legend_representatives():
    frame = data()
    fig = chart(frame, "exact_assignment_pct", "method", "Point", 3, False)
    page = standalone(fig, frame, "assignment_across_chunks")
    assert "const seen = new Set()" in page
    assert "Plotly.restyle(gd, {visible, showlegend})" in page


def test_multiple_configurations_have_distinct_readable_labels():
    frame = data().query("policy == 'stochastic'").copy()
    frame["scenario_count"] = "20"
    frame["scenario_seed"] = "42"
    frame["temperature"] = "1.0"
    extra = frame.copy()
    extra["scenario_count"] = "50"
    extra["policy_configuration_id"] = "stochastic-other"
    frame = pd.concat([frame, extra])
    fig = chart(frame, "exact_assignment_pct", "method", "Point", 3, False)
    assert sum(bool(t.showlegend) for t in fig.data) == 4
    assert len({t.name for t in fig.data}) == 4
    assert any("W=50" in t.name for t in fig.data)


@pytest.mark.parametrize("mode", ["Point", "Box"])
def test_ordinary_views_preserve_profiles_policies_and_sampling_roles(mode):
    from patient_assignment.proxy_policy_plots import ordinary_figures

    base = data()
    base["solver_status"] = "OPTIMAL"
    base["benchmark_id"] = "proxy"
    base["objective"] = "complexity-range"
    base["proxy_regret"] = 0.5
    base["is_seed_sample"] = True
    base["is_chunk_evaluation"] = True
    frames = []
    for profile in ["balanced", "unbalanced", "highly_unbalanced"]:
        part = base.copy()
        part["profile_id"] = profile
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    extra = frame.copy()
    extra["selection_seed"] = 43
    extra["is_chunk_evaluation"] = False
    frame = pd.concat([frame, extra], ignore_index=True)
    resolved = {
        "profiles": ["balanced", "unbalanced", "highly_unbalanced"],
        "chunk_seed": 42,
        "benchmark_id": "proxy",
        "objective": "complexity-range",
    }
    figures = ordinary_figures(frame, resolved, mode, 3)
    assert set(figures) == {
        "regret_by_chunk",
        "assignment_by_chunk",
        "regret_across_chunks",
        "assignment_across_chunks",
    }
    for name, fig in figures.items():
        expected = len(frame) if "by_chunk" in name else len(frame) // 2
        assert sum(len(t.y) for t in fig.data) == expected
        assert {t.meta["profile"] for t in fig.data} == set(resolved["profiles"])
        assert len({t.name for t in fig.data}) == 6
        assert {t.marker.symbol for t in fig.data} == {"circle", "diamond", "square"}
        assert sum(bool(t.showlegend) for t in fig.data) == 6
        assert len(fig.layout.annotations) == (3 if "by_chunk" in name else 0)
