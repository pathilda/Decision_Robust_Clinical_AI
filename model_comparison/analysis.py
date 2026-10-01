"""Load model outputs and compute descriptive and pairwise agreement results."""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


DEFAULT_CATEGORIES = (
    "Easy to treatment",
    "Moderate to treatment",
    "severe to treatment",
)


@dataclass(frozen=True)
class LoadedPredictions:
    label: str
    path: Path
    predictions: pd.Series
    source_rows: int
    missing_category_rows: int
    sha256: str


@dataclass(frozen=True)
class ComparisonResults:
    distributions: pd.DataFrame
    exact_agreement: pd.DataFrame
    quadratic_weighted_kappa: pd.DataFrame
    pairwise_n: pd.DataFrame
    input_summary: pd.DataFrame
    analysis_client_count: int | None


def load_predictions(
    *,
    label: str,
    path: Path,
    id_column: str,
    category_column: str,
    categories: tuple[str, ...],
) -> LoadedPredictions:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {resolved}")
    frame = _read_table(resolved)
    missing_columns = [
        column
        for column in (id_column, category_column)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"{label}: missing required columns {missing_columns}")

    selected = frame[[id_column, category_column]].copy()
    selected[id_column] = selected[id_column].map(_cell_text)
    missing_ids = selected[id_column].eq("")
    if missing_ids.any():
        rows = (selected.index[missing_ids] + 2).tolist()[:20]
        raise ValueError(f"{label}: missing client IDs at source rows {rows}")
    duplicate_ids = selected[id_column].duplicated(keep=False)
    if duplicate_ids.any():
        examples = selected.loc[duplicate_ids, id_column].drop_duplicates().tolist()[:20]
        raise ValueError(f"{label}: duplicate client IDs in results: {examples}")

    selected[category_column] = selected[category_column].map(_cell_text)
    missing_category_rows = int(selected[category_column].eq("").sum())
    selected = selected.loc[selected[category_column].ne("")].copy()
    invalid = sorted(set(selected[category_column]) - set(categories))
    if invalid:
        raise ValueError(
            f"{label}: categories not present in the configured ordered categories: {invalid}"
        )
    predictions = selected.set_index(id_column)[category_column]
    predictions.index.name = id_column
    predictions.name = label
    if predictions.empty:
        raise ValueError(f"{label}: no valid predictions remain after validation")
    return LoadedPredictions(
        label=label,
        path=resolved,
        predictions=predictions,
        source_rows=len(frame),
        missing_category_rows=missing_category_rows,
        sha256=_file_sha256(resolved),
    )


def compare_predictions(
    inputs: list[LoadedPredictions],
    *,
    categories: tuple[str, ...],
    cohort: str = "intersection",
) -> ComparisonResults:
    if len(inputs) < 2:
        raise ValueError("At least two model outputs are required")
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("Provide at least two unique ordered categories")
    labels = [item.label for item in inputs]
    if len(set(labels)) != len(labels):
        raise ValueError("Every input label must be unique")
    if cohort not in {"intersection", "pairwise"}:
        raise ValueError("cohort must be 'intersection' or 'pairwise'")

    series = {item.label: item.predictions for item in inputs}
    analysis_client_count: int | None = None
    if cohort == "intersection":
        common_ids = set.intersection(*(set(values.index) for values in series.values()))
        if not common_ids:
            raise ValueError("The model outputs have no client IDs in common")
        ordered_common_ids = [
            client_id
            for client_id in inputs[0].predictions.index
            if client_id in common_ids
        ]
        series = {label: values.loc[ordered_common_ids] for label, values in series.items()}
        analysis_client_count = len(ordered_common_ids)

    distributions = _category_distributions(series, categories)
    exact, kappa, pairwise_n = _pairwise_matrices(series, categories)
    input_summary = pd.DataFrame(
        [
            {
                "Configuration": item.label,
                "Path": str(item.path),
                "Source rows": item.source_rows,
                "Valid predictions": len(item.predictions),
                "Missing category rows": item.missing_category_rows,
                "SHA256": item.sha256,
            }
            for item in inputs
        ]
    )
    return ComparisonResults(
        distributions=distributions,
        exact_agreement=exact,
        quadratic_weighted_kappa=kappa,
        pairwise_n=pairwise_n,
        input_summary=input_summary,
        analysis_client_count=analysis_client_count,
    )


def _category_distributions(
    series: dict[str, pd.Series],
    categories: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for label, predictions in series.items():
        counts = predictions.value_counts().reindex(categories, fill_value=0)
        total = len(predictions)
        row: dict[str, int | float | str] = {"Configuration": label, "N": total}
        for category in categories:
            count = int(counts[category])
            row[f"{category} n"] = count
            row[f"{category} %"] = 100.0 * count / total if total else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_matrices(
    series: dict[str, pd.Series],
    categories: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = list(series)
    exact = pd.DataFrame(index=labels, columns=labels, dtype=float)
    kappa = pd.DataFrame(index=labels, columns=labels, dtype=float)
    pairwise_n = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for row_index, left_label in enumerate(labels):
        for column_index in range(row_index, len(labels)):
            right_label = labels[column_index]
            paired = pd.concat(
                [series[left_label].rename("left"), series[right_label].rename("right")],
                axis=1,
                join="inner",
            ).dropna()
            pair_count = len(paired)
            pairwise_n.loc[left_label, right_label] = pair_count
            pairwise_n.loc[right_label, left_label] = pair_count
            if left_label == right_label:
                agreement = 100.0
                weighted_kappa = 1.0
            elif pair_count == 0:
                agreement = math.nan
                weighted_kappa = math.nan
            else:
                agreement = 100.0 * float((paired["left"] == paired["right"]).mean())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    weighted_kappa = float(
                        cohen_kappa_score(
                            paired["left"],
                            paired["right"],
                            labels=list(categories),
                            weights="quadratic",
                        )
                    )
            exact.loc[left_label, right_label] = agreement
            exact.loc[right_label, left_label] = agreement
            kappa.loc[left_label, right_label] = weighted_kappa
            kappa.loc[right_label, left_label] = weighted_kappa
    exact.index.name = "Configuration"
    kappa.index.name = "Configuration"
    pairwise_n = pairwise_n.astype(int)
    pairwise_n.index.name = "Configuration"
    return exact, kappa, pairwise_n


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=object)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=object)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported prediction format {suffix!r}: {path}")


def _cell_text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
