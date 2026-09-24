from __future__ import annotations

import pandas as pd
import pytest

from clinical_target_extraction.src.data import DataConfig, inspect_input, load_notes


def make_config(tmp_path, rows: list[dict[str, object]], *, confirmed: bool = True) -> DataConfig:
    path = tmp_path / "notes.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Notes", index=False)
    return DataConfig(
        input_path=path,
        sheet_name="Notes",
        deidentified_confirmed=confirmed,
    )


def test_session_and_note_ids_are_derived_from_contiguous_row_order(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [
            {"ID": "C001", "statement2": "Client 1, session 1"},
            {"ID": "C001", "statement2": "Client 1, session 2"},
            {"ID": "C002", "statement2": "Client 2, session 1"},
            {"ID": "C002", "statement2": "Client 2, session 2"},
            {"ID": "C002", "statement2": "Client 2, session 3"},
        ],
    )
    notes = load_notes(config)
    report = inspect_input(config)

    assert [(note.client_id, note.session_index) for note in notes] == [
        ("C001", 1),
        ("C001", 2),
        ("C002", 1),
        ("C002", 2),
        ("C002", 3),
    ]
    assert [note.note_id for note in notes] == [
        "C001::S0001",
        "C001::S0002",
        "C002::S0001",
        "C002::S0002",
        "C002::S0003",
    ]
    assert report["actual_columns"] == ["ID", "statement2"]
    assert report["logical_column_mapping"] == {
        "client_id": "ID",
        "note_text": "statement2",
    }
    assert report["session_order"].startswith("derived from row order")


def test_client_id_cannot_reappear_in_a_later_block(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [
            {"ID": "C001", "statement2": "Client 1, session 1"},
            {"ID": "C002", "statement2": "Client 2, session 1"},
            {"ID": "C001", "statement2": "Client 1 incorrectly reappears"},
        ],
    )
    with pytest.raises(ValueError, match="one contiguous block"):
        load_notes(config)


@pytest.mark.parametrize("column", ["ID", "statement2"])
def test_missing_required_column_is_rejected(tmp_path, column: str) -> None:
    row = {"ID": "C001", "statement2": "A note"}
    del row[column]
    config = make_config(tmp_path, [row])
    with pytest.raises(ValueError, match="Configured columns not found"):
        load_notes(config)


def test_inference_requires_deidentification_confirmation(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [{"ID": "C001", "statement2": "A de-identified note"}],
        confirmed=False,
    )
    with pytest.raises(ValueError, match="deidentified_confirmed"):
        load_notes(config)
