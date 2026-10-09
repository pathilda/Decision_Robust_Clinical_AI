"""Synthetic cohort, input, CLI, and reporting acceptance cases (no solver)."""

import json
import random
import sys
from decimal import Decimal

import pandas as pd
import pytest

from patient_assignment.cli import main
from patient_assignment.cohort import (
    EMPLOYEE_COLUMN,
    ID_COLUMN,
    canonical_source,
    normalize_id,
    tags,
)
from patient_assignment.inputs import (
    WEIGHTS,
    check_note_hashes,
    load_membership,
    load_severity,
    parse_clinicians,
    preflight_capacity,
)
from patient_assignment.optimization import Optimum, SolverSettings, evaluate
from patient_assignment.reporting import comparison_summary
from patient_assignment.selection import coefficients, reference_regret


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Jane Smith SLP", {"SLP"}),
        ("PT Jane Smith", {"PT"}),
        ("Jane OT SLP", {"OT", "SLP"}),
        ("Jane  PT  Smith", {"PT"}),
        ("Jane slp", set()),
        ("Jane SLP,", set()),
        ("Jane (PT)", set()),
        ("ADOPT Smith", set()),
        (None, set()),
        ("Jane\tSLP", set()),
        ("Jane\nPT", set()),
    ],
)
def test_literal_tags(value, expected):
    assert tags(value) == expected


def test_first_row_union_no_filter_and_reporting():
    frame = pd.DataFrame(
        {
            ID_COLUMN: ["001A", "001A", "2B", "3C", "4"],
            EMPLOYEE_COLUMN: ["Jane OT", "Jane PT", "Name SLP", "PT OT", None],
        }
    )
    filtered, report = canonical_source(frame, disciplines=["PT", "SLP"])
    assert list(filtered[ID_COLUMN]) == ["2B", "3C"]
    assert report["duplicate_client_count"] == report["discarded_row_count"] == 1
    assert report["source_count"] == 4 and report["selected_count"] == 2
    assert report["recognized_tag_counts"] == {"PT": 1, "OT": 2, "SLP": 1}
    assert report["missing_empty_employee_count"] == 1
    assert len(canonical_source(frame.drop(columns=EMPLOYEE_COLUMN))[0]) == 4
    with pytest.raises(ValueError, match="employee column"):
        canonical_source(frame.drop(columns=EMPLOYEE_COLUMN), disciplines=["PT"])


@pytest.mark.parametrize(
    "value,expected", [(123.0, "123"), (" 001A ", "001A"), ("123.0", "123.0"), ("xB", "xB")]
)
def test_normalization(value, expected):
    assert normalize_id(value) == expected


@pytest.mark.parametrize("value", [None, "", "  ", float("nan"), float("inf"), float("-inf")])
def test_invalid_ids(value):
    with pytest.raises(ValueError):
        normalize_id(value)


def severity_file(tmp_path, ids=("001A", "2B"), categories=None, **extra):
    path = tmp_path / "severity.csv"
    pd.DataFrame(
        {
            ID_COLUMN: ids,
            "treatment_category": categories or ["Easy to treatment"] * len(ids),
            **extra,
        }
    ).to_csv(path, index=False)
    return path


def test_severity_preserves_ids_order_extra_rows_and_hash_check(tmp_path):
    path = severity_file(
        tmp_path,
        ids=["outside", "2B", "001A"],
        categories=["unknown", "severe to treatment", " Moderate to treatment "],
        input_sha256=["", "h2", "h1"],
    )
    a = load_severity(path, ["001A", "2B"])
    assert a.weights == {"001A": 2, "2B": 3} and a.metadata["ignored_rows"] == 1
    b = load_severity(path, ["2B", "001A"])
    assert a.weights == b.weights
    assert check_note_hashes(a, b)["verified_patient_count"] == 2
    b.note_hashes["001A"]["input_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        check_note_hashes(a, b)


@pytest.mark.parametrize(
    "categories", [["", "Easy to treatment"], ["Severe to treatment", "Easy to treatment"]]
)
def test_bad_selected_categories(tmp_path, categories):
    with pytest.raises(ValueError, match="category"):
        load_severity(severity_file(tmp_path, categories=categories), ["001A", "2B"])


def test_missing_failed_duplicate_predictions(tmp_path):
    with pytest.raises(ValueError, match="Missing selected"):
        load_severity(severity_file(tmp_path), ["001A", "missing"])
    with pytest.raises(ValueError, match="unsuccessful"):
        load_severity(severity_file(tmp_path, status=["completed", "failed"]), ["001A", "2B"])
    with pytest.raises(ValueError, match="unsuccessful"):
        load_severity(severity_file(tmp_path, success=[True, False]), ["001A", "2B"])
    with pytest.raises(ValueError, match="Duplicate normalized"):
        load_severity(severity_file(tmp_path, ids=["123", " 123 "]), ["123"])


def test_excel_numeric_collision_and_textual_zeros(tmp_path):
    path = tmp_path / "p.xlsx"
    pd.DataFrame(
        {ID_COLUMN: [123, "123"], "treatment_category": ["Easy to treatment"] * 2}
    ).to_excel(path, index=False)
    with pytest.raises(ValueError, match="collisions"):
        load_severity(path, ["123"])
    pd.DataFrame(
        {ID_COLUMN: ["001A", "001", "2B", 123.0], "treatment_category": ["Easy to treatment"] * 4}
    ).to_excel(path, index=False)
    assert set(load_severity(path, ["001A", "001", "2B", "123"]).weights) == {
        "001A",
        "001",
        "2B",
        "123",
    }


def test_chunk_reproducibility_remainder_and_no_global_rng_mutation(tmp_path):
    source = tmp_path / "source.xlsx"
    pd.DataFrame({ID_COLUMN: ["001A", "002B", "003C", "004", "005", "001A"]}).to_excel(
        source, index=False
    )
    arguments = ["chunk", "--input", str(source), "--chunk-size", "2", "--seed", "42"]
    state = random.getstate()
    for name in ("one", "two"):
        assert main(arguments + ["--output", str(tmp_path / name)]) == 0
    assert random.getstate() == state
    assert (tmp_path / "one/chunks.csv").read_bytes() == (tmp_path / "two/chunks.csv").read_bytes()
    chunks = load_membership(tmp_path / "one/chunks.csv")
    assert [len(ids) for ids in chunks.values()] == [2, 2, 1]
    assert len({p for ids in chunks.values() for p in ids}) == 5
    config = json.loads((tmp_path / "one/chunk_config.json").read_text())
    assert config["discarded_row_count"] == 1 and config["seed"] == 42
    assert main(arguments + ["--output", str(tmp_path / "one")]) == 2


def test_empty_chunk_nonpositive_and_membership_duplicates(tmp_path):
    source = tmp_path / "s.csv"
    pd.DataFrame({ID_COLUMN: ["001"], EMPLOYEE_COLUMN: ["OT"]}).to_csv(source, index=False)
    assert (
        main(
            [
                "chunk",
                "--input",
                str(source),
                "--chunk-size",
                "0",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "chunk",
                "--input",
                str(source),
                "--chunk-size",
                "1",
                "--discipline",
                "PT",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "out").exists()
    membership = tmp_path / "m.csv"
    pd.DataFrame({"patient_id": ["001", " 001 "], "chunk_id": ["c1", "c2"]}).to_csv(
        membership, index=False
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_membership(membership)


def test_operational_validation_and_exact_initial_decimal():
    clinicians = parse_clinicians([["A", "2", "3.1234567890123456789"], ["B", "0", "0"]], 3)
    assert str(clinicians[0].initial_severity) == "3.1234567890123456789"
    preflight_capacity({"small": ["1", "2", "3", "4"]}, clinicians, 3)
    with pytest.raises(ValueError, match="Infeasible"):
        preflight_capacity({"big": list("12345")}, clinicians, 3)
    for values in (
        [["A", "4", "4"]],
        [["A", "1", "NaN"]],
        [["A", "1.1", "2"]],
        [["A", "0", "0"], ["A", "0", "0"]],
    ):
        with pytest.raises(ValueError):
            parse_clinicians(values, 3)
    for settings in (
        SolverSettings(0),
        SolverSettings(float("nan")),
        SolverSettings(threads=0),
        SolverSettings(seed=-1),
    ):
        with pytest.raises(ValueError):
            settings.validate()


def test_reference_evaluation_explicit_allocations():
    clinicians = parse_clinicians([["A", "0", "0"], ["B", "0", "0"]], 2)
    patients = list("1234")
    reference = dict(zip(patients, [1, 1, 3, 3]))
    optimum = Optimum(
        {"stage1": {"optimal": True}, "stage2": {"optimal": True}},
        count_range=0,
        severity_range=Decimal(0),
    )
    bad = dict(zip(patients, "AABB"))
    good = dict(zip(patients, "ABAB"))
    assert reference_regret(bad, patients, clinicians, reference, 2, optimum)["raw_regret"] == "4"
    assert reference_regret(good, patients, clinicians, reference, 2, optimum)["raw_regret"] == "0"
    assert evaluate(good, patients, clinicians, reference, 2)[1:] == (0, Decimal(0))


def test_coefficients_partner_side_reference_and_order_independent():
    original, metadata = coefficients(42, "c", 1, "AI", ["2", "1"], ["B", "A"])
    reordered, second = coefficients(42, "c", 1, "AI", ["1", "2"], ["A", "B"])
    assert original == reordered and metadata == second
    coefficients(42, "different", 4, "other", ["3"], ["C"])
    assert coefficients(42, "c", 1, "AI", ["1", "2"], ["A", "B"])[0] == original
    assert coefficients(42, "c", 1, "Reference", ["1", "2"], ["A", "B"])[1] != metadata


def test_help_does_not_import_gpu_or_gurobi(capsys):
    import run

    before = set(sys.modules)
    for command in ("chunk", "solve", "compare-assignments"):
        with pytest.raises(SystemExit) as exit:
            run.main([command, "--help"])
        assert exit.value.code == 0
    assert not ({"gurobipy", "torch", "vllm", "transformers"} & (set(sys.modules) - before))


def test_summary_repeats_are_not_independent_chunks():
    rows = [
        {
            "chunk_id": "c1",
            "selection_certified": True,
            "patient_count": 4,
            "observed_disagreement_count": n,
            "observed_disagreement_rate": n / 4,
            "reported_regret": str(n),
        }
        for n in (0, 4)
    ]
    summary = comparison_summary(rows, {"c1": list("1234")}, "random", 2)
    assert summary["total_chunks"] == 1 and summary["certified_selections"] == 2
    assert (
        summary["mean_per_chunk_disagreement_rate"] == summary["pooled_patient_disagreement"] == 0.5
    )
    assert summary["within_chunk"][0]["regret"]["std"] is not None
    single = comparison_summary(rows[:1], {"c1": list("1234")}, "random", 1)
    assert single["within_chunk"][0]["regret"]["std"] is None


def test_inference_filters_first_row_before_limits_and_keeps_prompt_content(tmp_path):
    from clinical_target_extraction.src.cli import _selected_clients, build_parser
    from clinical_target_extraction.src.data import DataConfig, load_clients

    source = tmp_path / "source.xlsx"
    pd.DataFrame(
        {
            ID_COLUMN: ["001A", "001A", "002B", "003C"],
            EMPLOYEE_COLUMN: ["OT", "PT", "PT", "PT"],
            "SP text": ["first", "second", "selected", "more"],
            "assessment text": [""] * 4,
        }
    ).to_excel(source, index=False)
    clients = load_clients(
        DataConfig(input_path=source, deidentified_confirmed=True, disciplines=["PT"])
    )
    assert [c.client_id for c in clients] == ["002B", "003C"]
    assert clients[0].sp_text == "selected" and "PT" not in clients[0].note_text
    args = build_parser().parse_args(
        [
            "--pipeline",
            "classify",
            "--input",
            str(source),
            "--discipline",
            "PT",
            "--max-clients",
            "1",
        ]
    )
    assert _selected_clients(args, clients) == ["002B"]
    args.client_id = ["001A"]
    with pytest.raises(ValueError, match="Unknown"):
        _selected_clients(args, clients)


def test_standard_and_opendecider_share_numeric_id_normalization(tmp_path):
    from clinical_target_extraction.src.data import DataConfig, load_clients
    from opendecider_inference.cli import _data_config, build_parser

    path = tmp_path / "source.xlsx"
    pd.DataFrame(
        {
            ID_COLUMN: [123, "123", "001B"],
            EMPLOYEE_COLUMN: ["OT", "PT", "PT"],
            "SP text": ["first", "later", "keep"],
            "assessment text": [""] * 3,
        }
    ).to_excel(path, index=False)
    config = DataConfig(input_path=path, deidentified_confirmed=True)
    assert [c.client_id for c in load_clients(config)] == ["123", "001B"]
    args = build_parser().parse_args(
        ["--pipeline", "inspect", "--input", str(path), "--discipline", "PT"]
    )
    assert [c.client_id for c in load_clients(_data_config(args, confirmed=True))] == ["001B"]


def test_classification_comparison_filter_coverage_and_legacy_modes(tmp_path):
    from model_comparison.cli import main as compare_main

    source = tmp_path / "source.xlsx"
    pd.DataFrame(
        {ID_COLUMN: ["001A", "001A", "002B", "003C"], EMPLOYEE_COLUMN: ["OT", "PT", "PT", "SLP"]}
    ).to_excel(source, index=False)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame(
        {
            ID_COLUMN: ["001A", "002B", "003C"],
            "treatment_category": [
                "invalid outside selected cohort",
                "Easy to treatment",
                "severe to treatment",
            ],
        }
    ).to_csv(a, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: ["002B", "003C"],
            "treatment_category": ["Moderate to treatment", "severe to treatment"],
        }
    ).to_csv(b, index=False)
    for mode in ("intersection", "pairwise"):
        output = tmp_path / mode
        assert (
            compare_main(
                [
                    "--input",
                    f"A={a}",
                    "--input",
                    f"B={b}",
                    "--source-input",
                    str(source),
                    "--discipline",
                    "PT",
                    "--discipline",
                    "SLP",
                    "--cohort",
                    mode,
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        coverage = json.loads((output / "cohort_coverage.json").read_text())
        assert coverage["selected_count"] == 2 and coverage["discarded_row_count"] == 1
        assert coverage["prediction_coverage"]["A"]["excluded_by_source_cohort_rows"] == 1
        assert coverage["prediction_coverage"]["B"]["missing_selected_predictions"] == 0
    assert (
        compare_main(
            [
                "--input",
                f"A={a}",
                "--input",
                f"B={b}",
                "--discipline",
                "PT",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
        == 2
    )


def test_long_decimal_initial_state_is_preserved_beyond_default_precision():
    raw = "0.123456789012345678901234567890123456789"
    clinicians = parse_clinicians([["A", "0", raw], ["B", "0", "0"]], 2)
    loads, _dn, db = evaluate({"p": "A"}, ["p"], clinicians, {"p": 1}, 2)
    assert loads[0]["final_severity"] == "1.123456789012345678901234567890123456789"
    assert str(db) == loads[0]["final_severity"]
    assert clinicians[0].initial_severity_input == raw


def test_textual_na_ids_survive_excel_and_csv(tmp_path):
    for suffix in ("csv", "xlsx"):
        path = tmp_path / ("predictions." + suffix)
        frame = pd.DataFrame(
            {ID_COLUMN: ["NA", "001B"], "treatment_category": ["Easy to treatment"] * 2}
        )
        if suffix == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        assert set(load_severity(path, ["NA", "001B"]).weights) == {"NA", "001B"}


@pytest.mark.parametrize("backend", ["standard", "opendecider"])
def test_filtered_exports_exclude_unselected_historical_patients(tmp_path, backend):
    from clinical_target_extraction.src.classification_store import ClassificationStore
    from opendecider_inference.store import OpenDeciderStore

    store = (ClassificationStore if backend == "standard" else OpenDeciderStore)(tmp_path / backend)
    for patient in ("001A", "002B"):
        store.append_validated(
            {
                "client_id": patient,
                "model": backend,
                "treatment_category": "Easy to treatment",
                "source_row": 2,
                "input_sha256": "synthetic",
                "input_tokens": 1,
                "completed_at": "synthetic",
                "confidence": 1.0,
                "probabilities": {
                    category: float(category == "Easy to treatment") for category in WEIGHTS
                },
            }
        )
    store.materialize(["002B"])
    for filename in ("client_classifications.csv", "client_classifications.xlsx"):
        path = tmp_path / backend / filename
        frame = (
            pd.read_csv(path, dtype=object)
            if filename.endswith("csv")
            else pd.read_excel(path, dtype=object)
        )
        assert frame[ID_COLUMN].tolist() == ["002B"]
