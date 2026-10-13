"""Pooled agreement excludes provisional assignments and keeps repeats separate."""
import csv
from types import SimpleNamespace

from patient_assignment.reporting import write_patient_agreement


def test_patient_agreement_coverage_and_repeats(tmp_path, capsys):
    def row(patient, repeat, a, b, certified=True):
        return dict(patient_id=patient, repeat_id=repeat, selection_mode="random",
                    selection_certified=certified,
                    selected_clinician_A=a, selected_clinician_B=b)

    summaries = write_patient_agreement(
        tmp_path,
        [row("1", 1, "A", "A"), row("2", 1, "A", "B"),
         row("3", 1, "B", "B", False), row("1", 2, "B", "A")],
        {"c1": ["1", "2"], "c2": ["3"]}, "random", 2,
        [SimpleNamespace(name="A"), SimpleNamespace(name="B")],
        {"a": "QS", "b": "MS"},
    )
    assert summaries[0]["evaluated_patients"] == 2
    assert summaries[0]["same_clinician"] == 1
    assert summaries[0]["agreement_percent"] == 50
    assert not summaries[0]["coverage_complete"]
    assert summaries[1]["evaluated_patients"] == 1
    assert summaries[1]["agreement_percent"] == 0
    with (tmp_path / "clinician_agreement_matrix.csv").open() as f:
        matrix = list(csv.DictReader(f))
    assert matrix[0]["clinician_B:A"] == "1"
    assert matrix[0]["clinician_B:B"] == "1"
    assert "Incomplete coverage" in capsys.readouterr().out


def test_patient_agreement_no_certified_patients(tmp_path, capsys):
    summaries = write_patient_agreement(
        tmp_path, [], {"c": ["1"]}, "matched", 1,
        [SimpleNamespace(name="A")], {"a": "QS", "b": "MS"},
    )
    assert summaries[0]["agreement_percent"] is None
    assert summaries[0]["evaluated_patients"] == 0
    assert "unavailable" in capsys.readouterr().out
