from __future__ import annotations

import pandas as pd
import pytest

from clinical_target_extraction.src.data import DataConfig, inspect_input, load_clients


def make_config(tmp_path, rows: list[dict[str, object]], *, confirmed: bool = True) -> DataConfig:
    path = tmp_path / "clients.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Clients", index=False)
    return DataConfig(
        input_path=path,
        sheet_name="Clients",
        deidentified_confirmed=confirmed,
    )


def test_loads_one_row_per_client_and_combines_both_sources(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [
            {
                "Client AlayaCare Client ID": "C001",
                "SP text": "Needs weekly support.",
                "assessment text": "Moderate functional barriers.",
            },
            {
                "Client AlayaCare Client ID": "C002",
                "SP text": "Focused short-term plan.",
                "assessment text": "",
            },
        ],
    )

    clients = load_clients(config)
    report = inspect_input(config)

    assert [client.client_id for client in clients] == ["C001", "C002"]
    assert clients[0].source_row == 2
    assert "<SERVICE_PLANNING_TEXT>\nNeeds weekly support." in clients[0].note_text
    assert "<ASSESSMENT_TEXT>\nModerate functional barriers." in clients[0].note_text
    assert "[Not provided]" in clients[1].note_text
    assert report["logical_column_mapping"] == {
        "client_id": "Client AlayaCare Client ID",
        "sp_text": "SP text",
        "assessment_text": "assessment text",
    }
    assert report["one_row_per_client"] is True


def test_duplicate_client_ids_use_first_row_and_are_reported(tmp_path, capsys) -> None:
    config = make_config(
        tmp_path,
        [
            {
                "Client AlayaCare Client ID": "C001",
                "SP text": "One",
                "assessment text": "Two",
            },
            {
                "Client AlayaCare Client ID": "C001",
                "SP text": "Three",
                "assessment text": "Four",
            },
        ],
    )
    clients = load_clients(config)
    captured = capsys.readouterr()

    assert len(clients) == 1
    assert clients[0].sp_text == "One"
    assert "1 client has more than one row" in captured.err
    assert "ignoring 1 later row" in captured.err

    report = inspect_input(config)
    assert report["row_count"] == 2
    assert report["client_count"] == 1
    assert report["duplicate_client_count"] == 1
    assert report["ignored_later_row_count"] == 1


@pytest.mark.parametrize(
    "column", ["Client AlayaCare Client ID", "SP text", "assessment text"]
)
def test_missing_required_column_is_rejected(tmp_path, column: str) -> None:
    row = {
        "Client AlayaCare Client ID": "C001",
        "SP text": "One",
        "assessment text": "Two",
    }
    del row[column]
    with pytest.raises(ValueError, match="Configured columns not found"):
        load_clients(make_config(tmp_path, [row]))


def test_both_text_fields_cannot_be_blank(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [
            {
                "Client AlayaCare Client ID": "C001",
                "SP text": None,
                "assessment text": "  ",
            }
        ],
    )
    with pytest.raises(ValueError, match="cannot both be blank"):
        load_clients(config)


def test_inference_requires_deidentification_confirmation(tmp_path) -> None:
    config = make_config(
        tmp_path,
        [
            {
                "Client AlayaCare Client ID": "C001",
                "SP text": "One",
                "assessment text": "Two",
            }
        ],
        confirmed=False,
    )
    with pytest.raises(ValueError, match="deidentified_confirmed"):
        load_clients(config)
