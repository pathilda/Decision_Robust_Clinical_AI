"""Shared ID normalization and literal, first-row discipline selection."""

from __future__ import annotations

import math
import numbers

import pandas as pd

ID_COLUMN = "Client AlayaCare Client ID"
EMPLOYEE_COLUMN = "Employee Submission Full Name"
DISCIPLINES = ("PT", "OT", "SLP")
NORMALIZATION_POLICY = (
    "strip text; integral numeric cells to integer strings; preserve case/zeros/suffixes"
)
DUPLICATE_POLICY = "retain first source row per normalized ID; never merge notes or tags"


def normalize_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Missing client ID (blank/null)")
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Nonfinite client ID")
        if float(value).is_integer():
            return str(int(value))
    result = str(value).strip()
    if not result:
        raise ValueError("Missing client ID (blank/null)")
    return result


def tags(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    padded = " " + str(value) + " "
    return {tag for tag in DISCIPLINES if " " + tag + " " in padded}


def canonical_source(frame, id_column=ID_COLUMN, disciplines=(), employee_column=EMPLOYEE_COLUMN):
    if id_column not in frame:
        raise ValueError(f"Configured columns not found in input: [{id_column!r}]")
    if set(disciplines) - set(DISCIPLINES):
        raise ValueError("Discipline must be PT, OT, or SLP")
    if disciplines and employee_column not in frame:
        raise ValueError(f"Discipline filtering requires employee column {employee_column!r}")
    frame = frame.copy()
    frame[id_column] = frame[id_column].map(normalize_id)
    counts = frame[id_column].value_counts()
    repeated = counts[counts > 1]
    retained = frame.drop_duplicates(id_column, keep="first").copy()
    values = (
        retained[employee_column]
        if employee_column in retained
        else pd.Series(None, index=retained.index, dtype=object)
    )
    recognized = values.map(tags)
    selected = (
        recognized.map(lambda found: bool(found & set(disciplines)))
        if disciplines
        else pd.Series(True, index=retained.index)
    )
    report = {
        "source_row_count": len(frame),
        "source_count": len(retained),
        "selected_count": int(selected.sum()),
        "excluded_count": int((~selected).sum()),
        "missing_empty_employee_count": int(
            values.map(lambda v: pd.isna(v) or not str(v).strip()).sum()
        ),
        "recognized_tag_counts": {
            tag: int(recognized.map(lambda found, tag=tag: tag in found).sum())
            for tag in DISCIPLINES
        },
        "duplicate_client_count": len(repeated),
        "discarded_row_count": int((repeated - 1).sum()),
        "duplicate_policy": DUPLICATE_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
    }
    return retained.loc[selected].copy(), report


def add_filter_arguments(parser):
    parser.add_argument("--discipline", action="append", choices=DISCIPLINES, default=[])
    parser.add_argument("--employee-column", default=EMPLOYEE_COLUMN)
