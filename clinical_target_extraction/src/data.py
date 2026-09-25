"""Tabular input loading and pre-inference validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


SOURCE_LOGICAL_FIELDS = ("client_id", "note_text")


class ColumnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = "ID"
    note_text: str = "statement2"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: Path
    sheet_name: str | int | None = None
    deidentified_confirmed: bool = False
    columns: ColumnConfig = Field(default_factory=ColumnConfig)


@dataclass(frozen=True)
class ClinicalNote:
    client_id: str
    session_index: int
    note_id: str
    note_text: str


def read_input_table(config: DataConfig) -> pd.DataFrame:
    path = config.input_path
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=object)
    if suffix in {".xlsx", ".xls"}:
        sheet = config.sheet_name if config.sheet_name is not None else 0
        return pd.read_excel(path, sheet_name=sheet, dtype=object)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use CSV, Excel, or Parquet")


def load_notes(config: DataConfig, *, require_deidentified: bool = True) -> list[ClinicalNote]:
    if require_deidentified and not config.deidentified_confirmed:
        raise ValueError(
            "Refusing inference because data config does not set deidentified_confirmed: true"
        )
    frame = read_input_table(config)
    mapping = config.columns.model_dump()
    missing_columns = [actual for actual in mapping.values() if actual not in frame.columns]
    if missing_columns:
        raise ValueError(f"Configured columns not found in input: {missing_columns}")

    logical = frame[[mapping[name] for name in SOURCE_LOGICAL_FIELDS]].copy()
    logical.columns = list(SOURCE_LOGICAL_FIELDS)
    if logical.empty:
        raise ValueError("Input table contains no clinical notes")
    missing_rows: dict[str, list[int]] = {}
    for column in SOURCE_LOGICAL_FIELDS:
        mask = logical[column].isna()
        mask |= logical[column].astype(str).str.strip().eq("")
        if mask.any():
            missing_rows[column] = (logical.index[mask] + 2).astype(int).tolist()[:20]
    if missing_rows:
        raise ValueError(f"Missing required values (source row numbers, first 20): {missing_rows}")

    for column in SOURCE_LOGICAL_FIELDS:
        logical[column] = logical[column].astype(str).str.strip()

    _validate_contiguous_client_blocks(logical["client_id"].tolist())
    logical["session_index"] = (
        logical.groupby("client_id", sort=False).cumcount().astype(int) + 1
    )
    logical["note_id"] = [
        f"{client_id}::S{session_index:04d}"
        for client_id, session_index in zip(
            logical["client_id"], logical["session_index"], strict=True
        )
    ]

    return [
        ClinicalNote(
            client_id=row["client_id"],
            session_index=row["session_index"],
            note_id=row["note_id"],
            note_text=row["note_text"],
        )
        for row in logical.to_dict("records")
    ]


def _validate_contiguous_client_blocks(client_ids: list[str]) -> None:
    seen: set[str] = set()
    current: str | None = None
    for source_offset, client_id in enumerate(client_ids, start=2):
        if client_id == current:
            continue
        if client_id in seen:
            raise ValueError(
                f"Client {client_id!r} reappears at source row {source_offset}; "
                "each client's notes must occupy one contiguous block"
            )
        seen.add(client_id)
        current = client_id


def inspect_input(config: DataConfig) -> dict[str, Any]:
    frame = read_input_table(config)
    notes = load_notes(config, require_deidentified=False)
    clients = list(dict.fromkeys(note.client_id for note in notes))
    return {
        "input_path": str(config.input_path),
        "sheet_name": config.sheet_name,
        "actual_columns": [str(column) for column in frame.columns],
        "logical_column_mapping": config.columns.model_dump(),
        "session_order": "derived from row order within each contiguous client block",
        "generated_note_id_format": "{client_id}::S{session_index:04d}",
        "row_count": len(notes),
        "client_count": len(clients),
        "client_ids": clients,
        "deidentified_confirmed": config.deidentified_confirmed,
    }
