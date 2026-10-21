from copy import deepcopy

import pytest

from combine_proxy_results import comparable_settings


def settings():
    return {
        "chunks": {"01": ["a"]},
        "run_fingerprint": "first",
        "benchmark_seed": 1001,
        "probability_sources": {
            "Qwen": {
                "path": "probabilities.csv",
                "sha256": "full-file-hash",
                "normalized_sha256": "selected-patients-hash",
                "columns": {"easy": "prob_easy"},
                "id_column": "patient_id",
            }
        },
    }


def test_disjoint_cohort_hash_is_not_a_configuration_difference():
    first = settings()
    before = deepcopy(first)
    second = deepcopy(first)
    second["chunks"] = {"02": ["b"]}
    second["run_fingerprint"] = "second"
    second["probability_sources"]["Qwen"]["normalized_sha256"] = "other-patients"
    assert comparable_settings(first) == comparable_settings(second)
    assert first == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "different-file"),
        ("path", "different.csv"),
        ("columns", {"easy": "other"}),
        ("id_column", "other_id"),
    ],
)
def test_real_source_changes_remain_incompatible(field, value):
    first = settings()
    second = deepcopy(first)
    second["probability_sources"]["Qwen"][field] = value
    assert comparable_settings(first) != comparable_settings(second)


def test_experiment_settings_remain_checked():
    first = settings()
    second = deepcopy(first)
    second["benchmark_seed"] += 1
    assert comparable_settings(first) != comparable_settings(second)
