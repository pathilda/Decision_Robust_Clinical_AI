"""Excel input loading and validation for one-row-per-client classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


SOURCE_LOGICAL_FIELDS = ("client_id", "sp_text", "assessment_text")
CLIENT_ID_COLUMN = "Client AlayaCare Client ID"


class ColumnConfig(BaseModel):
    """Map the three logical fields to workbook column names."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = CLIENT_ID_COLUMN
    sp_text: str = "SP text"
    assessment_text: str = "assessment text"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: Path
    sheet_name: str | int | None = None
    deidentified_confirmed: bool = False
    columns: ColumnConfig = Field(default_factory=ColumnConfig)


@dataclass(frozen=True)
class ClientNote:
    """The single pre-treatment note submitted for one client."""

    client_id: str
    sp_text: str
    assessment_text: str
    note_text: str
    source_row: int


def read_input_table(config: DataConfig) -> pd.DataFrame:
    path = config.input_path
    if not path.exists():
        raise FileNotFoundError(f"Input workbook does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported input format {suffix!r}; use an Excel .xlsx or .xls file")
    sheet = config.sheet_name if config.sheet_name is not None else 0
    return pd.read_excel(path, sheet_name=sheet, dtype=object)


def load_clients(config: DataConfig, *, require_deidentified: bool = True) -> list[ClientNote]:
    """Validate the workbook and combine both text fields into one note per client."""

    if require_deidentified and not config.deidentified_confirmed:
        raise ValueError(
            "Refusing inference because data config does not set deidentified_confirmed: true"
        )

    frame = read_input_table(config)
    mapping = config.columns.model_dump()
    missing_columns = [actual for actual in mapping.values() if actual not in frame.columns]
    if missing_columns:
        raise ValueError(f"Configured columns not found in input: {missing_columns}")
    if frame.empty:
        raise ValueError("Input workbook contains no clients")

    logical = frame[[mapping[name] for name in SOURCE_LOGICAL_FIELDS]].copy()
    logical.columns = list(SOURCE_LOGICAL_FIELDS)
    logical["source_row"] = logical.index.to_series().astype(int) + 2

    client_ids = logical["client_id"].map(_cell_text)
    missing_id_rows = logical.loc[client_ids.eq(""), "source_row"].astype(int).tolist()[:20]
    if missing_id_rows:
        raise ValueError(f"Missing client ID values (source rows, first 20): {missing_id_rows}")
    logical["client_id"] = client_ids

    duplicate_mask = logical["client_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            logical.loc[duplicate_mask, ["client_id", "source_row"]]
            .groupby("client_id", sort=False)["source_row"]
            .apply(lambda values: [int(value) for value in values])
            .to_dict()
        )
        raise ValueError(
            "Each client must occupy exactly one row; duplicate client IDs found: "
            f"{duplicate_rows}"
        )

    for column in ("sp_text", "assessment_text"):
        logical[column] = logical[column].map(_cell_text)
    both_blank = logical["sp_text"].eq("") & logical["assessment_text"].eq("")
    if both_blank.any():
        rows = logical.loc[both_blank, "source_row"].astype(int).tolist()[:20]
        raise ValueError(
            "SP text and assessment text cannot both be blank "
            f"(source rows, first 20): {rows}"
        )

    return [
        ClientNote(
            client_id=str(row["client_id"]),
            sp_text=str(row["sp_text"]),
            assessment_text=str(row["assessment_text"]),
            note_text=combine_client_texts(
                sp_text=str(row["sp_text"]),
                assessment_text=str(row["assessment_text"]),
            ),
            source_row=int(row["source_row"]),
        )
        for row in logical.to_dict("records")
    ]


def combine_client_texts(*, sp_text: str, assessment_text: str) -> str:
    """Create one source-labeled note without altering either input text."""

    sp_value = sp_text if sp_text else "[Not provided]"
    assessment_value = assessment_text if assessment_text else "[Not provided]"
    return (
        "<SERVICE_PLANNING_TEXT>\n"
        f"{sp_value}\n"
        "</SERVICE_PLANNING_TEXT>\n\n"
        "<ASSESSMENT_TEXT>\n"
        f"{assessment_value}\n"
        "</ASSESSMENT_TEXT>"
    )


def inspect_input(config: DataConfig) -> dict[str, Any]:
    frame = read_input_table(config)
    clients = load_clients(config, require_deidentified=False)
    return {
        "input_path": str(config.input_path),
        "sheet_name": config.sheet_name,
        "actual_columns": [str(column) for column in frame.columns],
        "logical_column_mapping": config.columns.model_dump(),
        "row_count": len(clients),
        "client_count": len(clients),
        "client_ids": [client.client_id for client in clients],
        "one_row_per_client": True,
        "combined_note_sections": ["SP text", "assessment text"],
        "deidentified_confirmed": config.deidentified_confirmed,
    }


def _cell_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
