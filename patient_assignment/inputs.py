"""Strict membership/severity loaders and operational preflight, without Gurobi."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from .cohort import ID_COLUMN, normalize_id

WEIGHTS = {"Easy to treatment": 1, "Moderate to treatment": 2, "severe to treatment": 3}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path, sheet=None, columns=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype=object,
            keep_default_na=False,
            usecols=(lambda name: name in columns) if columns is not None else None,
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheet = int(sheet) if isinstance(sheet, str) and sheet.isdigit() else sheet
        return pd.read_excel(
            path,
            sheet_name=0 if sheet is None else sheet,
            dtype=object,
            keep_default_na=False,
            usecols=(lambda name: name in columns) if columns is not None else None,
        )
    raise ValueError("Use a .csv, .xlsx, or .xls input")


def unique_ids(frame, column):
    if column not in frame:
        raise ValueError(f"Missing ID column {column!r}")
    frame = frame.copy()
    frame[column] = frame[column].map(normalize_id)
    duplicates = frame.loc[frame[column].duplicated(False), column].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate normalized IDs / normalization collisions: {duplicates[:20]}")
    return frame


def load_membership(path):
    frame = unique_ids(read_table(path), "patient_id")
    if "chunk_id" not in frame:
        raise ValueError("Membership requires chunk_id")
    frame["chunk_id"] = frame["chunk_id"].map(normalize_id)
    if frame.empty:
        raise ValueError("Membership contains no patients")
    return {
        chunk: sorted(group.patient_id) for chunk, group in frame.groupby("chunk_id", sort=True)
    }


@dataclass
class SeverityInput:
    categories: dict[str, str]
    weights: dict[str, int]
    note_hashes: dict[str, dict[str, str]]
    metadata: dict


def load_severity(
    path, patients, id_column=ID_COLUMN, category_column="treatment_category", sheet=None
):
    frame = unique_ids(read_table(path, sheet), id_column)
    if category_column not in frame:
        raise ValueError(f"Missing category column {category_column!r}")
    selected_ids = set(patients)
    selected = frame.loc[frame[id_column].isin(selected_ids)].copy()
    missing = selected_ids - set(selected[id_column])
    if missing:
        raise ValueError(f"Missing selected predictions: {sorted(missing)[:20]}")
    categories, hashes = {}, {}
    for row in selected.to_dict("records"):
        patient = row[id_column]
        category = str(row[category_column]).strip() if pd.notna(row[category_column]) else ""
        if category not in WEIGHTS:
            raise ValueError(
                f"Invalid/blank category for selected patient {patient!r}: {category!r}"
            )
        for column in ("success", "successful", "status", "classification_status"):
            if column in row:
                value = str(row[column]).strip().lower()
                allowed = (
                    {"true", "1", "1.0"}
                    if column in {"success", "successful"}
                    else {"success", "successful", "completed", "validated", "ok"}
                )
                if value not in allowed:
                    raise ValueError(
                        f"Selected patient {patient!r} has unsuccessful {column}: {value!r}"
                    )
        categories[patient] = category
        hashes[patient] = {
            key: str(row[key]).strip()
            for key in ("input_sha256", "sp_text_sha256", "assessment_text_sha256")
            if key in row and pd.notna(row[key]) and str(row[key]).strip()
        }
    return SeverityInput(
        categories,
        {p: WEIGHTS[c] for p, c in categories.items()},
        hashes,
        {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": file_hash(path),
            "id_column": id_column,
            "category_column": category_column,
            "sheet": sheet,
            "source_rows": len(frame),
            "selected_rows": len(selected),
            "ignored_rows": len(frame) - len(selected),
        },
    )


def check_note_hashes(a, b):
    verified = 0
    for patient in a.weights:
        common = set(a.note_hashes[patient]) & set(b.note_hashes[patient])
        for key in common:
            if a.note_hashes[patient][key] != b.note_hashes[patient][key]:
                raise ValueError(f"Note hash mismatch for {patient!r} ({key})")
        if "input_sha256" in common or {"sp_text_sha256", "assessment_text_sha256"} <= common:
            verified += 1
    return {
        "verified_patient_count": verified,
        "selected_patient_count": len(a.weights),
        "note_consistency": "verified"
        if verified == len(a.weights)
        else "not fully verifiable; file hashes do not prove note identity",
    }


@dataclass(frozen=True)
class Clinician:
    name: str
    initial_count: int
    initial_severity: Decimal
    initial_severity_input: str = ""


def parse_clinicians(values, capacity):
    if capacity < 1:
        raise ValueError("Clinician capacity must be positive")
    clinicians = []
    for name, raw_count, raw_load in values:
        name = name.strip()
        try:
            count, load = int(raw_count), Decimal(raw_load)
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(
                "Clinician count must be an integer and severity a finite decimal"
            ) from exc
        if not name or count < 0 or count > capacity or not load.is_finite() or load < 0:
            raise ValueError(
                "Clinician name must be nonblank, 0 <= initial count <= capacity, and severity finite/nonnegative"
            )
        clinicians.append(Clinician(name, count, load, raw_load))
    if not clinicians or len({c.name for c in clinicians}) != len(clinicians):
        raise ValueError("Supply at least one clinician with unique names")
    return clinicians


def preflight_capacity(chunks, clinicians, capacity):
    remaining = sum(capacity - c.initial_count for c in clinicians)
    infeasible = {chunk: len(ids) for chunk, ids in chunks.items() if len(ids) > remaining}
    if infeasible:
        raise ValueError(
            f"Infeasible chunks (no truncation): {infeasible}; remaining capacity={remaining}"
        )
