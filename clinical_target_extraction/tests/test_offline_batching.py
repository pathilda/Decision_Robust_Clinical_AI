from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_target_extraction.src.data import ClinicalNote
from clinical_target_extraction.src.model_client import CompletionResult, ModelConfig
from clinical_target_extraction.src.run_extraction import process_clients


class BatchFakeClient:
    batches: list[list[tuple[str, int]]] = []

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete_batch(self, requests) -> list[CompletionResult]:
        batch: list[tuple[str, int]] = []
        results: list[CompletionResult] = []
        for _system, user_prompt, _schema, _schema_name in requests:
            client_id = re.search(r"CLIENT_ID: ([^\n]+)", user_prompt).group(1)
            current = re.search(r"CURRENT_SESSION_INDEX: (\d+)", user_prompt)
            session_index = int(current.group(1)) if current else 1
            batch.append((client_id, session_index))
            if session_index == 1:
                note_id = re.search(r"NOTE_ID: ([^\n]+)", user_prompt).group(1)
                payload = {
                    "client_id": client_id,
                    "session_index": 1,
                    "note_id": note_id,
                    "targets": [],
                }
            else:
                note_id = re.search(r"CURRENT_NOTE_ID: ([^\n]+)", user_prompt).group(1)
                payload = {
                    "client_id": client_id,
                    "session_index": session_index,
                    "note_id": note_id,
                    "existing_targets": [],
                    "new_targets": [],
                }
            results.append(CompletionResult(json.dumps(payload), 10))
        self.batches.append(batch)
        return results

    def close(self) -> None:
        pass


def test_direct_batches_preserve_each_clients_session_order(tmp_path: Path, monkeypatch) -> None:
    BatchFakeClient.batches = []
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_extraction.VLLMClient",
        BatchFakeClient,
    )
    notes = [
        ClinicalNote("C001", 1, "C001::S0001", "first"),
        ClinicalNote("C001", 2, "C001::S0002", "second"),
        ClinicalNote("C002", 1, "C002::S0001", "first"),
        ClinicalNote("C002", 2, "C002::S0002", "second"),
    ]
    config = ModelConfig(
        model_label="qwen",
        model_id="unused",
        tokenizer_id="unused",
        max_tokens=100,
        max_model_len=1000,
        max_num_seqs=2,
    )

    result = process_clients(
        notes=notes,
        selected_clients=["C001", "C002"],
        model_config=config,
        output_root=tmp_path,
        batch_size=2,
    )

    assert result == {"completed": 4, "reused": 0, "failed": 0, "context_overflow": 0}
    assert BatchFakeClient.batches == [
        [("C001", 1), ("C002", 1)],
        [("C001", 2), ("C002", 2)],
    ]
    assert (tmp_path / "qwen" / "session_targets.csv").exists()
    assert (tmp_path / "qwen" / "target_changes.csv").exists()


def test_new_target_is_assigned_and_carried_in_next_profile(tmp_path: Path, monkeypatch) -> None:
    class TargetFakeClient(BatchFakeClient):
        def complete_batch(self, requests) -> list[CompletionResult]:
            prompt = requests[0][1]
            current = re.search(r"CURRENT_SESSION_INDEX: (\d+)", prompt)
            session_index = int(current.group(1)) if current else 1
            if session_index == 1:
                payload = {
                    "client_id": "C001",
                    "session_index": 1,
                    "note_id": "C001::S0001",
                    "targets": [],
                }
            else:
                payload = {
                    "client_id": "C001",
                    "session_index": 2,
                    "note_id": "C001::S0002",
                    "existing_targets": [],
                    "new_targets": [{
                        "proposed_label": "Answering where questions",
                        "definition": "Answering questions about location.",
                        "aliases_in_note": ["where questions"],
                        "verbatim_evidence": ["where questions"],
                        "substantively_treated": 1,
                        "performance_observed": 1,
                    }],
                }
            return [CompletionResult(json.dumps(payload), 10)]

    monkeypatch.setattr(
        "clinical_target_extraction.src.run_extraction.VLLMClient",
        TargetFakeClient,
    )
    notes = [
        ClinicalNote("C001", 1, "C001::S0001", "first"),
        ClinicalNote("C001", 2, "C001::S0002", "Practiced where questions."),
    ]
    config = ModelConfig(
        model_label="qwen",
        model_id="unused",
        tokenizer_id="unused",
        max_tokens=100,
        max_model_len=1000,
        max_num_seqs=1,
    )
    process_clients(
        notes=notes,
        selected_clients=["C001"],
        model_config=config,
        output_root=tmp_path,
        batch_size=1,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "qwen" / "validated_sessions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and json.loads(line).get("record_type") == "validated"
    ]
    assert rows[-1]["session_profile"]["targets"][0]["target_id"] == "T001"
    assert rows[-1]["session_profile"]["targets"][0]["newly_added"] == 1
