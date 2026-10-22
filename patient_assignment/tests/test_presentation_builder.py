import hashlib
import re

import pandas as pd
import pytest

from build_presentation import OPTIONAL, PLOTS, TEMPLATE, assignment_scatter, build, resolve_plots


def plot_dir(tmp_path, assignment=True):
    root = tmp_path / "report"
    (root / "figures").mkdir(parents=True)
    for key, (stem, _) in (PLOTS | OPTIONAL).items():
        if key == "assignment_qwk" and not assignment:
            continue
        name = stem + ("_Box" if key.endswith("across") else "") + ".html"
        (root / "figures" / name).write_text("<html>saved plot " + key + "</html>")
    return root


def test_four_plot_source_build_preserves_existing_files(tmp_path, monkeypatch):
    root = plot_dir(tmp_path)
    monkeypatch.setattr(
        "build_presentation.subprocess.run", lambda *a, **k: pytest.fail("unexpected render")
    )
    output = tmp_path / "deck"
    manifest = build(root, output, prepare_only=True)
    assert set(manifest["plots"]) == {
        "regret_qwk",
        "assignment_qwk",
        "regret_across",
        "assignment_across",
    }
    assert not manifest["rendered"]
    text = (output / "slides.qmd").read_text(encoding="utf-8")
    assert "@@" not in text and "## Analysis III" not in text
    assert "regret_vs_exact_extraction" not in text
    assert text.count("<iframe ") == 4
    assert "What to do next?" in text
    assert len(re.findall(r"^## ", text, re.MULTILINE)) == 15  # plus title slide = 16
    for key, item in manifest["plots"].items():
        assert (
            hashlib.sha256((output / "plots" / f"{key}.html").read_bytes()).hexdigest()
            == item["sha256"]
        )
    assert (output / "source" / "Slides.tex").read_bytes() == (
        TEMPLATE / "source" / "Slides.tex"
    ).read_bytes()


def test_preserves_active_slide_titles_except_user_requested_changes(tmp_path):
    source = (TEMPLATE / "source" / "Slides.tex").read_text()
    source = re.sub(r"\\iffalse.*?\\fi\b", "", source, flags=re.DOTALL)
    titles = re.findall(r"\\begin\{frame\}\{([^}]+)\}", source)
    titles = [
        {"Conclusions": "Main results", "Next Possible Steps": "What to do next?"}.get(t, t)
        for t in titles
        if not t.startswith("Analysis III:")
    ]
    qmd = (TEMPLATE / "slides.qmd").read_text(encoding="utf-8")
    actual = re.findall(r"^## (.*?)(?: \{.*\})?$", qmd, re.MULTILINE)
    actual = [t.removeprefix("Analysis II: ") for t in actual if t != "Analysis roadmap"]
    assert actual == titles


def test_assignment_scatter_is_built_from_saved_outcomes(tmp_path):
    root = plot_dir(tmp_path, assignment=False)
    pd.DataFrame(
        [
            {
                "model_id": "Qwen",
                "profile_id": "balanced",
                "policy": "stochastic",
                "chunk_id": "01",
                "selection_seed": 42,
                "proxy_regret": 9,
                "exact_assignment_pct": 75,
                "solver_status": "OPTIMAL",
            }
        ]
    ).to_csv(root / "results.csv", index=False)
    # A partial policy schema should still fail rather than invent a config identity.
    frame = pd.read_csv(root / "results.csv")
    frame["policy_configuration_id"] = "saa20"
    frame.to_csv(root / "results.csv", index=False)
    pd.DataFrame([{"model_id": "Qwen", "qwk": 0.8}]).to_csv(
        root / "extraction_qwk.csv", index=False
    )
    page, hashes = assignment_scatter(root)
    assert "Exact assignment agreement (%)" in page
    assert '"y":[75]' in page or '"y":[75.0]' in page
    assert len(hashes) == 2
    manifest = build(root, tmp_path / "deck", prepare_only=True)
    assert "generated_from_saved_data" in manifest["plots"]["assignment_qwk"]


def test_missing_data_fails_before_output(tmp_path):
    root = plot_dir(tmp_path, assignment=False)
    with pytest.raises(ValueError, match="assignment_vs_qwk"):
        build(root, tmp_path / "deck", prepare_only=True)
    assert not (tmp_path / "deck").exists()


def test_rejects_ambiguous_plots_and_escape_paths(tmp_path):
    root = plot_dir(tmp_path)
    (root / "regret_vs_qwk.html").write_text("duplicate")
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_plots(root)
    with pytest.raises(ValueError, match="within"):
        resolve_plots(root, overrides={"regret_qwk": "../elsewhere.html"})


def test_output_is_not_overwritten(tmp_path):
    root = plot_dir(tmp_path)
    output = tmp_path / "deck"
    output.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        build(root, output, prepare_only=True)


def test_requested_title_cleanup_and_roadmap_location():
    text = (TEMPLATE / "slides.qmd").read_text(encoding="utf-8")
    assert "author:" not in text and "date:" not in text
    second = text.split("## Why this question matters")[1].split("## ")[0]
    assert "\u2014" not in second and "---" not in second
    assert (
        text.index("## From LLM assessments")
        < text.index("## Analysis roadmap")
        < text.index("## Three distinct lenses")
    )
    assert text.count("data-roadmap-title=") == 14
    assert 'aria-describedby="roadmap-detail"' in text
