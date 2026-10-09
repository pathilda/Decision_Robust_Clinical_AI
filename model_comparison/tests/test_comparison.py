from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from model_comparison.analysis import (
    DEFAULT_CATEGORIES,
    compare_predictions,
    load_predictions,
)
from model_comparison.cli import parse_input_specs
from model_comparison.report import write_report


ID_COLUMN = "Client AlayaCare Client ID"


def write_predictions(path: Path, categories: list[str], ids: list[str] | None = None) -> None:
    client_ids = ids or [f"C{index + 1:03d}" for index in range(len(categories))]
    pd.DataFrame(
        {
            ID_COLUMN: client_ids,
            "treatment_category": categories,
        }
    ).to_csv(path, index=False)


def load(label: str, path: Path):
    return load_predictions(
        label=label,
        path=path,
        id_column=ID_COLUMN,
        category_column="treatment_category",
        categories=DEFAULT_CATEGORIES,
    )


def test_distribution_exact_agreement_and_quadratic_kappa(tmp_path: Path) -> None:
    easy, moderate, severe = DEFAULT_CATEGORIES
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_predictions(first, [easy, moderate, severe, severe])
    write_predictions(second, [easy, severe, severe, moderate])

    results = compare_predictions(
        [load("QS", first), load("QR", second)],
        categories=DEFAULT_CATEGORIES,
    )

    assert results.analysis_client_count == 4
    assert results.exact_agreement.loc["QS", "QR"] == 50.0
    assert results.quadratic_weighted_kappa.loc["QS", "QR"] == pytest.approx(
        0.6363636
    )
    assert results.pairwise_n.loc["QS", "QR"] == 4
    distribution = results.distributions.set_index("Configuration")
    assert distribution.loc["QS", f"{severe} n"] == 2
    assert distribution.loc["QR", f"{moderate} %"] == 25.0


def test_intersection_cohort_uses_same_clients_for_every_result(tmp_path: Path) -> None:
    easy, moderate, severe = DEFAULT_CATEGORIES
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_predictions(first, [easy, moderate, severe], ["C001", "C002", "C003"])
    write_predictions(second, [moderate, severe], ["C002", "C003"])

    results = compare_predictions(
        [load("A", first), load("B", second)],
        categories=DEFAULT_CATEGORIES,
        cohort="intersection",
    )

    assert results.analysis_client_count == 2
    assert results.distributions["N"].tolist() == [2, 2]
    assert results.pairwise_n.to_numpy().tolist() == [[2, 2], [2, 2]]


def test_duplicate_client_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    write_predictions(
        path,
        [DEFAULT_CATEGORIES[0], DEFAULT_CATEGORIES[1]],
        ["C001", "C001"],
    )
    with pytest.raises(ValueError, match="duplicate client IDs"):
        load("duplicate", path)


def test_report_writes_tables_and_figures(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_predictions(first, list(DEFAULT_CATEGORIES))
    write_predictions(second, list(reversed(DEFAULT_CATEGORIES)))
    results = compare_predictions(
        [load("A", first), load("B", second)],
        categories=DEFAULT_CATEGORIES,
    )

    outputs = write_report(
        results,
        output_dir=tmp_path / "report",
        categories=DEFAULT_CATEGORIES,
        cohort="intersection",
        id_column=ID_COLUMN,
        category_column="treatment_category",
        title="Test comparison",
    )

    assert len(outputs) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_named_input_syntax_is_general_and_ordered() -> None:
    specs = parse_input_specs(["QS=/results/qwen.csv", "OD=/results/open.csv"])
    assert [label for label, _ in specs] == ["QS", "OD"]

    with pytest.raises(ValueError, match="Duplicate input label"):
        parse_input_specs(["QS=a.csv", "QS=b.csv"])
