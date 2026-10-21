from pathlib import Path

import pandas as pd
import pytest

from patient_assignment.inputs import file_hash
from proxy_extraction_tables import pooled_accuracy, write_tables


def cohort(tmp_path):
    inputs, hashes = {}, {}
    for model, labels in [
        ("model", ["easy", "easy", "severe"]),
        ("proxy", ["easy", "moderate", "severe"]),
    ]:
        path = tmp_path / f"{model}.csv"
        pd.DataFrame({"id": ["a", "b", "c"], "category": labels}).to_csv(path, index=False)
        inputs[model] = {
            "id": model,
            "path": str(path),
            "id_column": "id",
            "value_column": "category",
            "value_type": "category",
        }
        hashes[str(path)] = file_hash(path)
    return {
        "models": ["model"],
        "benchmark_id": "proxy",
        "inputs": inputs,
        "input_hashes": hashes,
        "chunks": {"01": ["a", "b"], "02": ["b", "c"]},
        "category_weights": {"easy": 1, "moderate": 2, "severe": 3},
        "category_order": ["easy", "moderate", "severe"],
        "category_aliases": {},
    }


def test_pooled_unique_clients_and_separate_tables(tmp_path):
    result = pooled_accuracy(cohort(tmp_path))
    assert result.iloc[0].n_patients == 3
    assert result.iloc[0].exact_accuracy_pct == pytest.approx(200 / 3)
    assert result.iloc[0].qwk == pytest.approx(0.8)
    page = write_tables(result, tmp_path)
    assert "Exact extraction agreement" in page
    for name in ["extraction_exact_agreement", "extraction_qwk"]:
        assert (tmp_path / f"{name}.csv").is_file()
        assert (tmp_path / f"{name}.html").is_file()


@pytest.mark.parametrize("changed", [True, False])
def test_missing_or_changed_sources_rejected(tmp_path, changed):
    resolved = cohort(tmp_path)
    path = Path(resolved["inputs"]["model"]["path"])
    if changed:
        path.write_text("modified")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        pooled_accuracy(resolved)


def test_undefined_qwk_is_explicit(tmp_path):
    resolved = cohort(tmp_path)
    resolved["chunks"] = {"01": ["a"]}
    result = pooled_accuracy(resolved)
    assert result.iloc[0].qwk is None
    assert "undefined" in result.iloc[0].qwk_status
