"""Common-success preprocessing, independently of optimization and inference."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from patient_assignment.proxy_chunking import run
from patient_assignment.proxy_cli import main, parser

DEMO = Path(__file__).resolve().parents[2] / "configs/proxy_uncertainty_demo"


def arguments(tmp_path, extra=()):
    source = tmp_path / "inputs"
    if not source.exists():
        shutil.copytree(DEMO, source)
    return parser().parse_args(
        [
            "proxy-chunks",
            "--config",
            str(source / "inputs.yaml"),
            "--models",
            "Qwen",
            "MedGemma",
            "OpenDecider",
            "--benchmark",
            "proxy",
            "--require-probabilities",
            "--uncertainty-config",
            str(source / "pilot.yaml"),
            "--chunk-size",
            "3",
            "--seed",
            "42",
            "--output",
            str(tmp_path / "chunks"),
            *extra,
        ]
    )


def save(frame, path):
    frame.to_csv(path, index=False)


def test_intersection_before_chunking_and_reproducibility(tmp_path):
    args = arguments(tmp_path)
    source = args.config.parent
    frame = pd.read_csv(source / "qwen.csv", dtype=str)
    save(frame[frame.patient_id != "001"], source / "qwen.csv")
    frame = pd.read_csv(source / "medgemma.csv", dtype=str)
    frame["status"] = "success"
    frame.loc[frame.patient_id == "002", "status"] = "failed"
    save(frame, source / "medgemma.csv")
    frame = pd.read_csv(source / "opendecider.csv", dtype=str)
    frame.loc[frame.patient_id == "003", "probability_easy"] = "NaN"
    save(frame, source / "opendecider.csv")
    frame = pd.read_csv(source / "proxy.csv", dtype=str)
    save(frame[frame.patient_id != "004"], source / "proxy.csv")
    assert run(args) == 0
    chunks = pd.read_csv(args.output / "chunks.csv", dtype=str)
    assert set(chunks.patient_id) == {"005", "006", "007", "008"}
    assert sorted(chunks.groupby("chunk_id").size()) == [1, 3]
    exclusions = pd.read_csv(args.output / "exclusions.csv", dtype=str)
    assert set(exclusions.patient_id) == {"001", "002", "003", "004"}
    original = (args.output / "chunks.csv").read_bytes()
    args.output = tmp_path / "repeat"
    assert run(args) == 0
    assert original == (args.output / "chunks.csv").read_bytes()
    args.output = tmp_path / "drop"
    args.drop_remainder = True
    assert run(args) == 0
    report = json.loads((args.output / "cohort_report.json").read_text())
    assert report["common_valid_clients"] == 4
    assert report["included_clients"] == 3
    assert report["dropped_remainder"] == 1


def test_optional_original_population_filter_rechunks_ids(tmp_path):
    args = arguments(tmp_path)
    source = args.config.parent
    whitelist = tmp_path / "whitelist.csv"
    pd.DataFrame({"patient_id": ["001", "002", "004", "008"], "chunk_id": ["old"] * 4}).to_csv(
        whitelist, index=False
    )
    args.eligible_chunks = whitelist
    assert run(args) == 0
    frame = pd.read_csv(args.output / "chunks.csv", dtype=str)
    assert set(frame.patient_id) == {"001", "002", "004", "008"}
    assert "old" not in set(frame.chunk_id)
    coverage = pd.read_csv(args.output / "model_coverage.csv")
    assert not coverage.loc[coverage.model_id == "proxy", "probabilities_required"].item()
    assert source.exists()


def test_separate_probability_failures_and_missing_ids(tmp_path):
    args = arguments(tmp_path)
    path = args.config.parent / "medgemma_probabilities.csv"
    frame = pd.read_csv(path, dtype=str)
    frame = frame[frame.subject != "001"]
    frame["success"] = "true"
    frame.loc[frame.subject == "002", "success"] = "false"
    save(frame, path)
    assert run(args) == 0
    common = pd.read_csv(args.output / "common_successful_clients.csv", dtype=str)
    assert len(common) == 6
    assert not {"001", "002"} & set(common.patient_id)


def test_duplicate_prediction_ids_are_not_silently_intersected(tmp_path):
    args = arguments(tmp_path)
    path = args.config.parent / "qwen.csv"
    frame = pd.read_csv(path, dtype=str)
    save(pd.concat([frame, frame.iloc[:1]]), path)
    with pytest.raises(ValueError, match="Duplicate"):
        run(args)
    assert not args.output.exists()


def test_missing_probability_columns_are_actionable(tmp_path):
    args = arguments(tmp_path)
    path = args.config.parent / "qwen.csv"
    frame = pd.read_csv(path, dtype=str).drop(columns=["probability_easy"])
    save(frame, path)
    with pytest.raises(ValueError, match="missing probability columns"):
        run(args)
    args.require_probabilities = False
    args.uncertainty_config = None
    assert run(args) == 0  # label-only intersection remains supported


def test_empty_cohort_and_empty_drop_are_rejected(tmp_path):
    args = arguments(tmp_path)
    args.chunk_size = 20
    args.drop_remainder = True
    with pytest.raises(ValueError, match="Fewer common"):
        run(args)
    assert not args.output.exists()
    path = args.config.parent / "qwen.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["status"] = "failed"
    save(frame, path)
    with pytest.raises(ValueError, match="No common"):
        run(args)


def test_cli_dispatch_and_membership_is_valid_for_analysis(tmp_path):
    import run as entry
    from patient_assignment.proxy_inputs import prepare
    from patient_assignment.tests.test_proxy_uncertainty import args_for

    args = arguments(tmp_path)
    assert (
        entry.main(
            [
                "proxy-chunks",
                "--config",
                str(args.config),
                "--models",
                "Qwen",
                "MedGemma",
                "OpenDecider",
                "--benchmark",
                "proxy",
                "--require-probabilities",
                "--chunk-size",
                "3",
                "--output",
                str(args.output),
            ]
        )
        == 0
    )
    analysis = args_for(tmp_path)
    analysis.config = args.config
    analysis.chunks = args.output / "chunks.csv"
    assert sum(len(p) for p in prepare(analysis)["resolved"]["chunks"].values()) == 8
    assert (
        main(
            [
                "proxy-chunks",
                "--config",
                str(args.config),
                "--models",
                "Qwen",
                "--benchmark",
                "proxy",
                "--chunk-size",
                "0",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
        == 2
    )
