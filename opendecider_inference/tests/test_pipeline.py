from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinical_target_extraction.src.data import ClientNote
from opendecider_inference.local_model import LocalOpenDecider, _build_adapter_overlay
from opendecider_inference.pipeline import process_clients, validate_response
from opendecider_inference.rubric import CATEGORIES, QUESTION_NAME


def client(client_id: str, row: int) -> ClientNote:
    return ClientNote(
        client_id=client_id,
        sp_text=f"SP for {client_id}",
        assessment_text=f"Assessment for {client_id}",
        note_text=(
            f"<SERVICE_PLANNING_TEXT>\nSP for {client_id}\n"
            "</SERVICE_PLANNING_TEXT>\n\n"
            f"<ASSESSMENT_TEXT>\nAssessment for {client_id}\n"
            "</ASSESSMENT_TEXT>"
        ),
        source_row=row,
    )


def model_directories(tmp_path: Path) -> tuple[Path, Path]:
    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "opendecider.json").write_text(
        '{"kind":"small","base_model":"remote/name"}\n',
        encoding="utf-8",
    )
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    return adapter, base


class FakeOpenDecider:
    constructions = 0
    batches: list[int] = []

    def __init__(self, adapter_path: Path, base_path: Path, *, device: str) -> None:
        assert adapter_path.is_dir()
        assert base_path.is_dir()
        assert device == "cuda"
        type(self).constructions += 1

    def classify_batch(self, states, questions):
        assert set(questions[QUESTION_NAME]["criteria"]) == set(CATEGORIES)
        type(self).batches.append(len(states))
        return [
            {
                "answers": {
                    QUESTION_NAME: {
                        "choice": "Moderate to treatment",
                        "confidence": 0.7,
                        "probabilities": {
                            "Easy to treatment": 0.1,
                            "Moderate to treatment": 0.7,
                            "severe to treatment": 0.2,
                        },
                    }
                },
                "usage": {"input_tokens": 25, "output_tokens": 0},
            }
            for _ in states
        ]

    def close(self) -> None:
        pass


def test_pipeline_batches_and_writes_comparable_results(tmp_path: Path) -> None:
    FakeOpenDecider.constructions = 0
    FakeOpenDecider.batches = []
    adapter, base = model_directories(tmp_path)
    clients = [client("C001", 2), client("C002", 3), client("C003", 4)]

    result = process_clients(
        clients=clients,
        selected_clients=[item.client_id for item in clients],
        adapter_path=adapter,
        base_path=base,
        output_root=tmp_path / "results",
        batch_size=2,
        model_factory=FakeOpenDecider,
    )

    assert result == {"completed": 3, "reused": 0, "failed": 0}
    assert FakeOpenDecider.batches == [2, 1]
    output = pd.read_excel(
        tmp_path / "results" / "opendecider" / "client_classifications.xlsx"
    )
    assert output["Client AlayaCare Client ID"].tolist() == ["C001", "C002", "C003"]
    assert output["treatment_category"].tolist() == ["Moderate to treatment"] * 3
    assert output["probability_moderate"].tolist() == [0.7, 0.7, 0.7]


def test_unchanged_results_are_reused_without_loading_model(tmp_path: Path) -> None:
    FakeOpenDecider.constructions = 0
    FakeOpenDecider.batches = []
    adapter, base = model_directories(tmp_path)
    clients = [client("C001", 2)]
    arguments = dict(
        clients=clients,
        selected_clients=["C001"],
        adapter_path=adapter,
        base_path=base,
        output_root=tmp_path / "results",
        model_factory=FakeOpenDecider,
    )

    process_clients(**arguments)
    result = process_clients(**arguments)

    assert result == {"completed": 0, "reused": 1, "failed": 0}
    assert FakeOpenDecider.constructions == 1


def test_response_validation_rejects_unknown_category() -> None:
    response = {
        "answers": {
            QUESTION_NAME: {
                "choice": "easy",
                "probabilities": {
                    "Easy to treatment": 0.7,
                    "Moderate to treatment": 0.2,
                    "severe to treatment": 0.1,
                },
            }
        }
    }
    with pytest.raises(ValueError, match="Unsupported treatment category"):
        validate_response(response)


def test_adapter_overlay_uses_local_base_without_modifying_adapter(tmp_path: Path) -> None:
    adapter, base = model_directories(tmp_path)
    (adapter / "adapter_model.safetensors").write_bytes(b"test")
    overlay = tmp_path / "overlay"
    overlay.mkdir()

    _build_adapter_overlay(source=adapter, destination=overlay, base_path=base)

    original = (adapter / "opendecider.json").read_text(encoding="utf-8")
    rewritten = (overlay / "opendecider.json").read_text(encoding="utf-8")
    assert '"base_model":"remote/name"' in original
    assert str(base) in rewritten
    assert (overlay / "adapter_model.safetensors").resolve() == (
        adapter / "adapter_model.safetensors"
    ).resolve()


def test_local_model_falls_back_when_batch_api_is_unavailable() -> None:
    class OldOpenDecider:
        def system_one(self, state, questions):
            return {"state": state, "questions": questions}

    wrapper = LocalOpenDecider.__new__(LocalOpenDecider)
    wrapper.model = OldOpenDecider()
    questions = {"decision": {"type": "choice"}}

    assert wrapper.classify_batch(["one", "two"], questions) == [
        {"state": "one", "questions": questions},
        {"state": "two", "questions": questions},
    ]
