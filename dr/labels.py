"""Exploratory analysis of the clinician-labelled clients and their ICF coverage.

    python -m dr.labels --labels-file /path/labels.xlsx --outputs-dir /scratch/me/dr_outputs

Prints:
1. File checks: rows, unique clients, duplicate / blank IDs, missing classes or confidences.
2-4. All labelled clients: class distribution (counts and %), confidence distribution (counts
   and %; summary statistics when numeric) and the class x confidence table.
5. ICF coverage: for each model, how many labelled clients have a successful ICF profile,
   a failed one, or none; how many have one for all models.
6-8. The same three tables for the overlap: clients with a successful ICF from every model.

Writes <outputs-dir>/labels/label_coverage.csv (one row per labelled client: ID, class,
confidence, ICF status per model). Clinical text columns are never read into the outputs.
Client IDs are normalized exactly as the ICF step does (leading zeros kept).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from patient_assignment.cohort import normalize_id
from patient_assignment.inputs import read_table

from .common import MODEL_KEYS, JsonlStore, Workspace, classes_config

DEFAULT_ID_COLUMN = "Client AlayaCare Client ID"


def find_column(frame, given, pattern, what):
    """Use the given column, else the single column whose name matches ``pattern``."""
    if given:
        if given not in frame.columns:
            raise ValueError(f"{what} column {given!r} not found; columns: {list(frame.columns)}")
        return given
    matches = [c for c in frame.columns if re.search(pattern, str(c), re.IGNORECASE)]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot pick the {what} column automatically (matches: {matches}); pass it with "
            f"--{what}-column. Columns: {list(frame.columns)}"
        )
    return matches[0]


def blank(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def as_class(value):
    """4, 4.0, "4" or "Class 4" -> 4; anything else -> None."""
    match = re.fullmatch(r"(?:class\s*)?(\d+(?:\.0+)?)", str(value).strip(), re.IGNORECASE)
    return int(float(match.group(1))) if match else None


CONFIDENCE_ORDER = ("very low", "low", "mid", "medium", "moderate", "high", "very high")


def confidence_rank(value):
    """Order text confidences low -> high ("Low Confidence" < "Mid Confidence" < ...)."""
    text = re.sub(r"\s*confidence\s*", " ", str(value), flags=re.IGNORECASE).strip().lower()
    return (CONFIDENCE_ORDER.index(text), "") if text in CONFIDENCE_ORDER else (99, str(value))


def distribution(series, order):
    """Counts and % in ``order``; levels with no clients are shown as 0."""
    counts = series.value_counts(dropna=False)
    counts = counts.reindex(list(order) + [v for v in counts.index if v not in order], fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame({"n": counts, "pct": (100 * counts / total).round(1) if total else counts})


def icf_status(ws, model, ids):
    path = ws.icf(model)
    if not path.is_file():
        return None
    store = JsonlStore(path)
    by_client = {}
    for record in store.records.values():
        by_client[str(record.get("client_id", record["key"]))] = record
    return {
        cid: ("ok" if by_client[cid].get("parse_ok") else "failed") if cid in by_client else "none"
        for cid in ids
    }


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def class_names(labels):
    valid = set(classes_config()["ids"])
    return labels["class"].where(labels["class"].isin(valid)).map(
        lambda v: "missing/invalid" if pd.isna(v) else f"Class {int(v)}").rename("class")


def levels(labels):
    """Display order of classes and confidences, fixed from the full label set so the overlap
    tables list the same rows and columns (zeros included)."""
    classes = [f"Class {i}" for i in sorted(classes_config()["ids"])]
    if (class_names(labels) == "missing/invalid").any():
        classes.append("missing/invalid")
    values = labels["confidence"].dropna().unique()
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").notna().all() and len(values)
    confidences = sorted(values, key=float if numeric else confidence_rank)
    if labels["confidence"].isna().any():
        confidences.append("missing")
    return classes, [str(c) for c in confidences], bool(numeric)


def describe(labels, title, first_number, order):
    """Class distribution, confidence distribution and class x confidence for ``labels``."""
    class_order, confidence_order, numeric_confidence = order
    classes = class_names(labels)
    confidence = labels["confidence"].fillna("missing").astype(str).rename("confidence")
    n = first_number

    section(f"{n}. CLASS DISTRIBUTION - {title}")
    print(distribution(classes, class_order).to_string())

    section(f"{n + 1}. CONFIDENCE DISTRIBUTION - {title}")
    print(distribution(confidence, confidence_order).to_string())
    numeric = pd.to_numeric(labels["confidence"], errors="coerce")
    if numeric_confidence and numeric.notna().any():
        stats = numeric.describe()
        print(f"\nnumeric: mean {stats['mean']:.2f}, median {numeric.median():.2f}, "
              f"min {stats['min']:g}, max {stats['max']:g}, sd {stats['std']:.2f}")
        print("\nmean confidence by class:")
        print(numeric.groupby(classes).agg(["count", "mean"]).round(2).to_string())

    section(f"{n + 2}. CLASS x CONFIDENCE (counts) - {title}")
    table = pd.crosstab(classes, confidence).reindex(
        index=class_order, columns=confidence_order, fill_value=0)
    table.loc["total"] = table.sum()
    table["total"] = table.sum(axis=1)
    print(table.to_string())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels-file", type=Path, required=True, help="Excel/CSV of clinician labels")
    p.add_argument("--sheet", default=None, help="Sheet name or index (default: first sheet)")
    p.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    p.add_argument("--class-column", default=None, help="Default: the one column named like 'class'")
    p.add_argument("--confidence-column", default=None,
                   help="Default: the one column named like 'confidence'")
    p.add_argument("--outputs-dir", type=Path, required=True,
                   help="Holds <model>/icf/icf.jsonl; label_coverage.csv is written under labels/")
    p.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    args = p.parse_args(argv)
    ws = Workspace(outputs_dir=args.outputs_dir)

    frame = read_table(args.labels_file, args.sheet)
    id_col = find_column(frame, args.id_column, r"client.*id", "id")
    class_col = find_column(frame, args.class_column, r"class", "class")
    conf_col = find_column(frame, args.confidence_column, r"confiden", "confidence")
    print(f"Labels file: {args.labels_file}")
    print(f"Columns used: id={id_col!r}, class={class_col!r}, confidence={conf_col!r}")

    # ------------------------------------------------------------------ 1. file checks
    section("1. FILE CHECKS")
    blank_ids = frame[id_col].map(blank)
    rows = frame[~blank_ids].copy()
    rows["client_id"] = rows[id_col].map(normalize_id)
    duplicated = rows["client_id"][rows["client_id"].duplicated(keep=False)]
    print(f"Rows: {len(frame)}   blank IDs: {int(blank_ids.sum())}   "
          f"unique clients: {rows['client_id'].nunique()}")
    if len(duplicated):
        print(f"Duplicate IDs ({duplicated.nunique()} clients, {len(duplicated)} rows); "
              f"the first row of each is used: {sorted(set(duplicated))}")
    labels = rows.drop_duplicates("client_id", keep="first").reset_index(drop=True)
    labels["class"] = labels[class_col].map(lambda v: None if blank(v) else as_class(v))
    labels["confidence"] = labels[conf_col].map(lambda v: None if blank(v) else str(v).strip())
    valid = set(classes_config()["ids"])
    raw_class_missing = labels[class_col].map(blank)
    bad_class = labels[~raw_class_missing & ~labels["class"].isin(valid)]
    print(f"Missing class: {int(raw_class_missing.sum())}   "
          f"missing confidence: {int(labels['confidence'].isna().sum())}")
    if len(bad_class):
        print(f"Class values that are not classes {sorted(valid)}: "
              f"{bad_class[class_col].astype(str).tolist()}")

    order = levels(labels)
    describe(labels, "ALL LABELLED CLIENTS", 2, order)

    # ------------------------------------------------------------------ 5. ICF coverage
    section("5. ICF COVERAGE OF LABELLED CLIENTS")
    ids = list(labels["client_id"])
    coverage = labels[["client_id", "class", "confidence"]].copy()
    have = []
    for model in args.models:
        status = icf_status(ws, model, ids)
        if status is None:
            print(f"{model:14s} no ICF file at {ws.icf(model)}")
            continue
        coverage[f"icf_{model}"] = coverage["client_id"].map(status)
        counts = coverage[f"icf_{model}"].value_counts()
        print(f"{model:14s} ICF ok {counts.get('ok', 0):3d} / {len(ids)}   "
              f"failed {counts.get('failed', 0):3d}   not run {counts.get('none', 0):3d}")
        have.append(model)
    if have:
        all_ok = (coverage[[f"icf_{m}" for m in have]] == "ok").all(axis=1)
        coverage["icf_ok_all_models"] = all_ok
        print(f"\nICF ok for all {len(have)} models: {int(all_ok.sum())} / {len(ids)}")
        print("by class:")
        status = all_ok.map({True: "ICF all models", False: "missing"}).rename("icf")
        print(pd.crosstab(class_names(labels), status, margins=True,
                          margins_name="total").to_string())
        overlap = labels[all_ok.to_numpy()].reset_index(drop=True)
        describe(overlap, f"CLIENTS WITH ICF FROM ALL {len(have)} MODELS (n={len(overlap)})", 6,
                 order)

    out = ws.outputs / "labels" / "label_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out, index=False)
    print(f"\nPer-client table written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
