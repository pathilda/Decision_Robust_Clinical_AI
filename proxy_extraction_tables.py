"""Extraction agreement tables; distinct from allocation agreement and solver policies."""

from pathlib import Path

import pandas as pd

from patient_assignment.experiment_inputs import load_score
from patient_assignment.inputs import file_hash
from patient_assignment.proxy_inputs import agreement


def pooled_accuracy(resolved):
    patients = sorted({p for members in resolved["chunks"].values() for p in members})
    weights = resolved["category_weights"]
    scoring = {
        "category_weights": weights,
        "category_aliases": resolved["category_aliases"],
        "bounds": [min(weights.values()), max(weights.values())],
        "scale": "proxy complexity",
    }
    loaded = {}
    benchmark = resolved["benchmark_id"]
    for model in [*resolved["models"], benchmark]:
        entry = resolved["inputs"][model]
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(
                f"Combined extraction tables require the original inference file: {path}"
            )
        if file_hash(path) != resolved["input_hashes"].get(str(path)):
            raise ValueError(f"Inference file changed since analysis: {path}")
        data = load_score(entry, Path.cwd(), scoring)
        if set(patients) - data["categories"].keys():
            raise ValueError(f"Missing valid extraction labels for {model} in the combined cohort")
        loaded[model] = data["categories"]
    rows = []
    for model in resolved["models"]:
        exact, qwk, status = agreement(
            [loaded[model][p] for p in patients],
            [loaded[benchmark][p] for p in patients],
            resolved["category_order"],
        )
        rows.append(
            {
                "model_id": model,
                "benchmark_id": benchmark,
                "n_patients": len(patients),
                "exact_accuracy_pct": exact,
                "qwk": qwk,
                "qwk_status": status,
            }
        )
    return pd.DataFrame(rows)


def write_tables(accuracy, output):
    output = Path(output)
    accuracy = accuracy.copy()
    accuracy["qwk"] = pd.to_numeric(accuracy["qwk"], errors="coerce")
    common = ["model_id", "benchmark_id", "n_patients"]
    tables = (
        (
            "extraction_exact_agreement",
            "Exact extraction agreement with proxy (%)",
            accuracy[common + ["exact_accuracy_pct"]].rename(
                columns={"exact_accuracy_pct": "exact_agreement_pct"}
            ),
        ),
        (
            "extraction_qwk",
            "Extraction quadratic weighted kappa (QWK) with proxy",
            accuracy[common + ["qwk", "qwk_status"]],
        ),
    )
    sections = []
    for name, title, table in tables:
        table.to_csv(output / f"{name}.csv", index=False)
        content = f"<h2>{title}</h2>" + table.to_html(index=False, escape=True, na_rep="NA")
        content += f'<p><a href="{name}.csv">Download CSV</a></p>'
        (output / f"{name}.html").write_text(
            '<!doctype html><meta charset="utf-8">' + content, encoding="utf-8"
        )
        sections.append(content)
    return (
        "<p>Extraction categories compared once per unique client against the proxy; "
        "these are not clinician-assignment agreement scores. QWK uses the configured "
        "category order and is NA when undefined. Policies and selection seeds do not "
        "create additional extraction observations.</p>" + "".join(sections)
    )
