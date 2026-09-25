from __future__ import annotations

import json

from clinical_target_extraction.src.data import ClinicalNote
from clinical_target_extraction.src.output_store import OutputStore
from clinical_target_extraction.src.run_extraction import call_with_one_repair
from clinical_target_extraction.src.schemas import Session1Output
from clinical_target_extraction.src.validate import validate_session1


class FakeModelClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, *args: object, **kwargs: object) -> tuple[str, int]:
        self.calls += 1
        return next(self.responses), 100


def test_invalid_output_gets_exactly_one_successful_repair(tmp_path) -> None:
    note = ClinicalNote("C001", 1, "N001", "The client completed 8 of 10.")
    valid = {
        "client_id": "C001",
        "session_index": 1,
        "note_id": "N001",
        "targets": [{
            "proposed_label": "Following two-step directions",
            "definition": "Completing two sequential spoken directions.",
            "aliases_in_note": ["two-step directions"],
            "verbatim_evidence": ["completed 8 of 10"],
            "substantively_treated": 1,
            "performance_observed": 1,
        }],
    }
    fake = FakeModelClient(["not json", json.dumps(valid)])
    store = OutputStore(tmp_path)

    output, input_tokens = call_with_one_repair(
        model_client=fake,  # type: ignore[arg-type]
        store=store,
        system_prompt="system",
        task_prompt="task",
        schema=Session1Output,
        schema_name="Session1Output",
        validate_context=lambda value: validate_session1(value, note),
        audit_context={"client_id": "C001", "session_index": 1},
    )

    assert output.note_id == "N001"
    assert input_tokens == 100
    assert fake.calls == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["attempt"], row["valid"]) for row in rows] == [
        ("initial", False),
        ("repair", True),
    ]
