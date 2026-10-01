"""Command-line interface for reusable model-agreement comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clinical_target_extraction.src.data import CLIENT_ID_COLUMN

from .analysis import DEFAULT_CATEGORIES, compare_predictions, load_predictions
from .report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordinal classifications from any number of model output files"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named CSV, Excel, or Parquet result; repeat for every configuration",
    )
    parser.add_argument("--output", type=Path, default=Path("comparison_results"))
    parser.add_argument("--id-column", default=CLIENT_ID_COLUMN)
    parser.add_argument("--category-column", default="treatment_category")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Ordered category; repeat from lowest to highest severity",
    )
    parser.add_argument(
        "--cohort",
        choices=["intersection", "pairwise"],
        default="intersection",
        help="Use one common client cohort or the available clients for each pair",
    )
    parser.add_argument("--title", default="Model agreement comparison")
    return parser


def parse_input_specs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --input {value!r}; expected LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"Invalid --input {value!r}; label and path are required")
        if label in labels:
            raise ValueError(f"Duplicate input label: {label!r}")
        labels.add(label)
        parsed.append((label, Path(raw_path)))
    if len(parsed) < 2:
        raise ValueError("Repeat --input for at least two model configurations")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        categories = tuple(args.categories or DEFAULT_CATEGORIES)
        inputs = [
            load_predictions(
                label=label,
                path=path,
                id_column=args.id_column,
                category_column=args.category_column,
                categories=categories,
            )
            for label, path in parse_input_specs(args.input)
        ]
        results = compare_predictions(inputs, categories=categories, cohort=args.cohort)
        outputs = write_report(
            results,
            output_dir=args.output,
            categories=categories,
            cohort=args.cohort,
            id_column=args.id_column,
            category_column=args.category_column,
            title=args.title,
        )
        print(f"Compared {len(inputs)} configurations")
        if results.analysis_client_count is not None:
            print(f"Common clients analyzed: {results.analysis_client_count}")
        print(f"Report directory: {args.output.expanduser().resolve()}")
        print(f"Files created: {len(outputs)}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
