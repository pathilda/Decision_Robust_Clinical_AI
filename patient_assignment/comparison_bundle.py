"""Run category agreement and both allocation comparisons on one strict cohort."""

from copy import copy

import pandas as pd

from .cohort import ID_COLUMN
from .inputs import WEIGHTS, file_hash
from .reporting import write_json


def run_all(args, output, chunks, labels, predictions):
    from model_comparison.analysis import LoadedPredictions, compare_predictions
    from model_comparison.report import write_report

    from .cli import execute_args

    categories = tuple(WEIGHTS)
    patients = [p for ids in chunks.values() for p in ids]
    inputs = []
    for side in ("a", "b"):
        path = getattr(args, "predictions_" + side).expanduser().resolve()
        inputs.append(LoadedPredictions(
            label=labels[side], path=path,
            predictions=pd.Series(
                [predictions[side].categories[p] for p in patients],
                index=pd.Index(patients, name=ID_COLUMN), name="treatment_category",
            ),
            source_rows=predictions[side].metadata["source_rows"],
            excluded_rows=predictions[side].metadata["ignored_rows"],
            missing_category_rows=0, sha256=file_hash(path),
        ))
    print("\n=== Extraction: treatment-category agreement across chunk membership ===", flush=True)
    results = compare_predictions(inputs, categories=categories, cohort="intersection")
    write_report(
        results, output_dir=output / "extraction", categories=categories,
        cohort="intersection", id_column=ID_COLUMN, category_column="treatment_category",
        title=f"{labels['a']} vs {labels['b']} treatment-category agreement",
    )
    write_json(output / "extraction" / "membership_config.json", {
        "cohort": "Union of saved chunks; no intersection-based patient exclusion",
        "membership_path": str(args.chunks.resolve()),
        "membership_sha256": file_hash(args.chunks),
        "membership_patients": len(patients),
        "prediction_inputs": {side: predictions[side].metadata for side in ("a", "b")},
    })
    print(f"Patients analyzed: {len(patients):,}")
    print(f"Exact category agreement: {results.exact_agreement.loc[labels['a'], labels['b']]:.2f}%")
    print("Quadratic weighted kappa: "
          f"{results.quadratic_weighted_kappa.loc[labels['a'], labels['b']]:.4f}")
    status = {
        "command": "compare-all",
        "cohort": "Union of saved chunks; identical patients for all three comparisons",
        "membership_patients": len(patients),
        "membership_sha256": file_hash(args.chunks),
        "options": vars(args),
        "stages": {"extraction": {"exit_code": 0, "output": str(output / "extraction")}},
        "complete": False,
    }
    # Preserve completed stages even if a later stage is interrupted.
    write_json(output / "comparison_bundle_status.json", status)
    for mode in ("matched", "random"):
        child = copy(args)
        child.command = "compare-assignments"
        child.selection_mode = mode
        child.selection_repeats = 1 if mode == "matched" else args.selection_repeats
        child.output = output / mode
        print(f"\n=== {mode.capitalize()} assignment comparison ===", flush=True)
        code = execute_args(child)
        status["stages"][mode] = {"exit_code": code, "output": str(child.output)}
        write_json(output / "comparison_bundle_status.json", status)
        # Time-limited matched results must not prevent running random comparison.
    codes = [stage["exit_code"] for stage in status["stages"].values()]
    status["complete"] = all(code == 0 for code in codes)
    status["exit_code"] = max(codes)
    write_json(output / "comparison_bundle_status.json", status)
    print(f"\nCombined reports: {output}", flush=True)
    return status["exit_code"]
