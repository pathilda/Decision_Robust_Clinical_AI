from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from clinical_target_extraction.src.data import ClientNote
from clinical_target_extraction.src.model_client import CompletionResult, ModelConfig
from clinical_target_extraction.src.run_classification import process_clients
from clinical_target_extraction.src.schemas import (
    ClassificationOutput,
    ClassificationWithBriefReasoningOutput,
)


def client(client_id: str, row: int) -> ClientNote:
    sp_text = f"SP information for {client_id}"
    assessment = f"Assessment information for {client_id}"
    return ClientNote(
        client_id=client_id,
        sp_text=sp_text,
        assessment_text=assessment,
        note_text=(
            f"<SERVICE_PLANNING_TEXT>\n{sp_text}\n</SERVICE_PLANNING_TEXT>\n\n"
            f"<ASSESSMENT_TEXT>\n{assessment}\n</ASSESSMENT_TEXT>"
        ),
        source_row=row,
    )


def model_config(batch_size: int = 2) -> ModelConfig:
    return ModelConfig(
        model_label="qwen",
        model_id="unused",
        tokenizer_id="unused",
        max_tokens=100,
        max_model_len=1000,
        max_num_seqs=batch_size,
    )


class BatchFakeClient:
    batches: list[list[str]] = []
    constructions = 0

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        type(self).constructions += 1

    def complete_batch(self, requests) -> list[CompletionResult]:
        ids = [re.search(r"CLIENT_ID: ([^\n]+)", request[1]).group(1) for request in requests]
        type(self).batches.append(ids)
        return [
            CompletionResult(
                json.dumps(
                    {
                        "client_id": client_id,
                        "treatment_category": "Moderate to treatment",
                    }
                ),
                20,
            )
            for client_id in ids
        ]

    def close(self) -> None:
        pass


def test_independent_clients_are_batched_and_written_to_excel(
    tmp_path: Path, monkeypatch
) -> None:
    BatchFakeClient.batches = []
    BatchFakeClient.constructions = 0
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", BatchFakeClient
    )
    clients = [client("C001", 2), client("C002", 3), client("C003", 4)]

    result = process_clients(
        clients=clients,
        selected_clients=[item.client_id for item in clients],
        model_config=model_config(),
        output_root=tmp_path,
        batch_size=2,
    )

    assert result == {"completed": 3, "reused": 0, "failed": 0, "context_overflow": 0}
    assert BatchFakeClient.batches == [["C001", "C002"], ["C003"]]
    output = pd.read_excel(tmp_path / "qwen" / "client_classifications.xlsx")
    assert output["Client AlayaCare Client ID"].tolist() == [
        "C001",
        "C002",
        "C003",
    ]
    assert output["treatment_category"].tolist() == ["Moderate to treatment"] * 3


def test_unchanged_clients_are_reused_without_loading_the_model(
    tmp_path: Path, monkeypatch
) -> None:
    BatchFakeClient.batches = []
    BatchFakeClient.constructions = 0
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", BatchFakeClient
    )
    clients = [client("C001", 2)]
    arguments = dict(
        clients=clients,
        selected_clients=["C001"],
        model_config=model_config(),
        output_root=tmp_path,
        batch_size=2,
    )

    process_clients(**arguments)
    result = process_clients(**arguments)

    assert result == {"completed": 0, "reused": 1, "failed": 0, "context_overflow": 0}
    assert BatchFakeClient.constructions == 1


def test_invalid_category_gets_one_structured_repair(tmp_path: Path, monkeypatch) -> None:
    class RepairFakeClient(BatchFakeClient):
        call_count = 0

        def complete_batch(self, requests) -> list[CompletionResult]:
            type(self).call_count += 1
            category = "easy" if self.call_count == 1 else "Easy to treatment"
            return [
                CompletionResult(
                    json.dumps(
                        {"client_id": "C001", "treatment_category": category}
                    ),
                    20,
                )
            ]

    RepairFakeClient.call_count = 0
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", RepairFakeClient
    )

    result = process_clients(
        clients=[client("C001", 2)],
        selected_clients=["C001"],
        model_config=model_config(1),
        output_root=tmp_path,
        batch_size=1,
    )

    assert result["completed"] == 1
    assert RepairFakeClient.call_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "qwen" / "raw_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["attempt"], row["valid"]) for row in rows] == [
        ("initial", False),
        ("repair", True),
    ]


def test_schema_accepts_only_the_three_exact_labels() -> None:
    for category in (
        "Easy to treatment",
        "Moderate to treatment",
        "severe to treatment",
    ):
        ClassificationOutput(client_id="C001", treatment_category=category)

    with pytest.raises(ValidationError):
        ClassificationOutput(client_id="C001", treatment_category="Easy to treat")


def test_brief_reasoning_prompt_writes_reasoning_to_separate_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class BriefReasoningFakeClient(BatchFakeClient):
        def complete_batch(self, requests) -> list[CompletionResult]:
            assert all(
                request[2] is ClassificationWithBriefReasoningOutput
                for request in requests
            )
            return [
                CompletionResult(
                    json.dumps(
                        {
                            "client_id": "C001",
                            "brief_reasoning": (
                                "The note documents multiple needs requiring coordination."
                            ),
                            "treatment_category": "Moderate to treatment",
                        }
                    ),
                    30,
                )
            ]

    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient",
        BriefReasoningFakeClient,
    )

    result = process_clients(
        clients=[client("C001", 2)],
        selected_clients=["C001"],
        model_config=model_config(1),
        output_root=tmp_path,
        batch_size=1,
        prompt_style="brief-reasoning",
    )

    assert result["completed"] == 1
    output = pd.read_excel(
        tmp_path / "qwen_brief_reasoning" / "client_classifications.xlsx"
    )
    assert output.loc[0, "prompt_style"] == "brief-reasoning"
    assert output.loc[0, "brief_reasoning"].startswith("The note documents")
    assert output.loc[0, "treatment_category"] == "Moderate to treatment"


def test_brief_reasoning_schema_requires_nonempty_reasoning() -> None:
    with pytest.raises(ValidationError):
        ClassificationWithBriefReasoningOutput(
            client_id="C001",
            brief_reasoning="",
            treatment_category="Easy to treatment",
        )
