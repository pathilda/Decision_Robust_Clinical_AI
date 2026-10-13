"""Combined comparison uses one strict cohort and runs both selection modes."""

import json

import pandas as pd
import pytest

from patient_assignment import cli
from patient_assignment.cohort import ID_COLUMN


def arguments(tmp_path):
    chunks = tmp_path / "chunks.csv"
    pd.DataFrame({"patient_id": ["001", "002"], "chunk_id": ["c1", "c2"]}).to_csv(
        chunks, index=False,
    )
    for name, categories in (
        ("a", ["Easy to treatment", "Moderate to treatment", "severe to treatment"]),
        ("b", ["Easy to treatment", "severe to treatment", "severe to treatment"]),
    ):
        pd.DataFrame({ID_COLUMN: ["001", "002", "outside"],
                      "treatment_category": categories}).to_csv(tmp_path / f"{name}.csv", index=False)
    return ["compare-all", "--chunks", str(chunks),
            "--predictions-a", str(tmp_path / "a.csv"), "--label-a", "QS",
            "--predictions-b", str(tmp_path / "b.csv"), "--label-b", "MS",
            "--clinician", "A", "0", "0", "--clinician-capacity", "3",
            "--threads", "20", "--time-limit", "300", "--selection-seed", "7",
            "--selection-repeats", "3", "--output", str(tmp_path / "reports")]


@pytest.mark.parametrize("matched_code", [0, 1, 2])
def test_bundle_cohort_outputs_and_child_options(tmp_path, monkeypatch, matched_code):
    args = cli.build_parser().parse_args(arguments(tmp_path))
    execute = cli.execute_args
    calls = []

    def child(child_args):
        calls.append(child_args)
        return matched_code if child_args.selection_mode == "matched" else 0

    monkeypatch.setattr(cli, "execute_args", child)
    assert execute(args) == matched_code
    assert [c.selection_mode for c in calls] == ["matched", "random"]
    assert [c.selection_repeats for c in calls] == [1, 3]
    assert all(c.threads == 20 and c.time_limit == 300 and c.selection_seed == 7 for c in calls)
    assert [c.output.name for c in calls] == ["matched", "random"]
    matrix = pd.read_csv(tmp_path / "reports/extraction/exact_agreement_matrix.csv", index_col=0)
    assert matrix.loc["QS", "MS"] == 50  # Outside patient must not contribute.
    n = pd.read_csv(tmp_path / "reports/extraction/pairwise_sample_size_matrix.csv", index_col=0)
    assert n.loc["QS", "MS"] == 2
    status = json.loads((tmp_path / "reports/comparison_bundle_status.json").read_text())
    assert status["complete"] == (matched_code == 0)
    assert status["stages"]["random"]["exit_code"] == 0


def test_bundle_rejects_missing_members_before_reports(tmp_path):
    argv = arguments(tmp_path)
    pd.DataFrame({ID_COLUMN: ["001"], "treatment_category": ["Easy to treatment"]}).to_csv(
        tmp_path / "b.csv", index=False,
    )
    assert cli.main(argv) == 2
    assert not (tmp_path / "reports").exists()


def test_bundle_help_and_defaults():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["compare-all", "--help"])
    assert exc.value.code == 0
