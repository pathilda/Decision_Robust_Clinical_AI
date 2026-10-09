"""Command-line interface for reusable model-agreement comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clinical_target_extraction.src.data import CLIENT_ID_COLUMN
from patient_assignment.cohort import add_filter_arguments, canonical_source
from patient_assignment.inputs import file_hash, read_table

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
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--source-sheet")
    parser.add_argument("--source-id-column", default=CLIENT_ID_COLUMN)
    add_filter_arguments(parser)
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
        eligible_ids = None
        coverage = {}
        if args.discipline and args.source_input is None:
            raise ValueError("--discipline requires --source-input")
        if args.source_input is not None:
            source, coverage = canonical_source(read_table(args.source_input, args.source_sheet, columns=[args.source_id_column, args.employee_column]), args.source_id_column, args.discipline, args.employee_column)
            eligible_ids = set(source[args.source_id_column])
            coverage.update(source_path=str(args.source_input.resolve()), source_sha256=file_hash(args.source_input))
        inputs = [
            load_predictions(
                label=label,
                path=path,
                id_column=args.id_column,
                category_column=args.category_column,
                categories=categories,
                eligible_ids=eligible_ids,
            )
            for label, path in parse_input_specs(args.input)
        ]
        results = compare_predictions(inputs, categories=categories, cohort=args.cohort)
        if eligible_ids is not None:
            coverage["prediction_coverage"] = {item.label: {
                "source_prediction_rows": item.source_rows,
                "selected_valid_predictions": len(item.predictions),
                "missing_selected_predictions": len(eligible_ids - set(item.predictions.index)),
                "excluded_or_missing_category_rows": item.source_rows - len(item.predictions),
                "excluded_by_source_cohort_rows": item.excluded_rows,
                "selected_missing_category_rows": item.missing_category_rows,
            } for item in inputs}
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
        if coverage:
            (args.output / "cohort_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        if results.analysis_client_count is not None:
            print(f"Common clients analyzed: {results.analysis_client_count}")
        print(f"Report directory: {args.output.expanduser().resolve()}")
        print(f"Files created: {len(outputs)}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
