"""Excel input loading and validation for one-row-per-client classification."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from patient_assignment.cohort import EMPLOYEE_COLUMN, canonical_source, normalize_id

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
    disciplines: list[str] = Field(default_factory=list)
    employee_column: str = EMPLOYEE_COLUMN


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
    return pd.read_excel(path, sheet_name=sheet, dtype=object, keep_default_na=False)


def load_clients(
    config: DataConfig, *, require_deidentified: bool = True, coverage_out: dict | None = None
) -> list[ClientNote]:
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

    # Canonicalize before filtering: notes and discipline always use the same first row.
    frame, coverage = canonical_source(
        frame, mapping["client_id"], config.disciplines, config.employee_column
    )
    if coverage_out is not None:
        coverage_out.update(coverage, disciplines=config.disciplines, employee_column=config.employee_column)
    if config.disciplines:
        print("Cohort coverage: " + json.dumps(coverage), file=sys.stderr)
    duplicate_client_count = coverage["duplicate_client_count"]
    ignored_later_row_count = coverage["discarded_row_count"]
    if duplicate_client_count:
        client_word = "client has" if duplicate_client_count == 1 else "clients have"
        row_word = "row" if ignored_later_row_count == 1 else "rows"
        print(
            f"warning: {duplicate_client_count} {client_word} more than one row; "
            f"using the first row for each and ignoring {ignored_later_row_count} later {row_word}",
            file=sys.stderr,
        )
    if frame.empty:
        raise ValueError("No patients remain in the selected cohort")
    logical = frame[[mapping[name] for name in SOURCE_LOGICAL_FIELDS]].copy()
    logical.columns = list(SOURCE_LOGICAL_FIELDS)
    logical["source_row"] = logical.index.to_series().astype(int) + 2

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
    mapping = config.columns.model_dump()
    if mapping["client_id"] in frame.columns:
        source_ids = frame[mapping["client_id"]].map(normalize_id)
        duplicate_client_count, ignored_later_row_count = _duplicate_stats(source_ids)
    else:
        duplicate_client_count = 0
        ignored_later_row_count = 0
    _, coverage = canonical_source(frame, mapping["client_id"], config.disciplines, config.employee_column)
    return {
        "cohort_coverage": coverage,
        "disciplines": config.disciplines,
        "input_path": str(config.input_path),
        "sheet_name": config.sheet_name,
        "actual_columns": [str(column) for column in frame.columns],
        "logical_column_mapping": config.columns.model_dump(),
        "row_count": len(frame),
        "client_count": len(clients),
        "client_ids": [client.client_id for client in clients],
        "one_row_per_client": duplicate_client_count == 0,
        "analysis_unit": "the first source row for each client",
        "duplicate_client_count": duplicate_client_count,
        "ignored_later_row_count": ignored_later_row_count,
        "duplicate_policy": "use the first source row for each client",
        "combined_note_sections": ["SP text", "assessment text"],
        "deidentified_confirmed": config.deidentified_confirmed,
    }


def _cell_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _duplicate_stats(client_ids: pd.Series) -> tuple[int, int]:
    counts = client_ids[client_ids.ne("")].value_counts(sort=False)
    repeated = counts[counts.gt(1)]
    return len(repeated), int((repeated - 1).sum())
