from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from clinical_target_extraction.src.data import ClinicalNote
from clinical_target_extraction.src.prompts import build_later_prompt, load_prompts
from clinical_target_extraction.src.registry import TargetRegistry
from clinical_target_extraction.src.schemas import (
    DiscoveredTarget,
    LaterSessionOutput,
    NewTargetCandidate,
    ProfileTarget,
    Session1Output,
    SessionProfile,
)
from clinical_target_extraction.src.validate import (
    OutputValidationError,
    audit_later_session_quality,
    audit_session1_quality,
    validate_later_session,
    validate_session1,
)


def discovered(*, treated: int = 1, observed: int = 1) -> DiscoveredTarget:
    return DiscoveredTarget(
        proposed_label="Following two-step directions",
        definition="Completing spoken directions containing two sequential steps.",
        aliases_in_note=["two-step directions"],
        verbatim_evidence=["completed 8 of 10"],
        substantively_treated=treated,
        performance_observed=observed,
    )


def previous_profile() -> SessionProfile:
    return SessionProfile(
        client_id="C001",
        session_index=1,
        note_id="N001",
        targets=[
            ProfileTarget(
                target_id="T001",
                canonical_label="Following two-step directions",
                definition="Completing spoken directions containing two sequential steps.",
                aliases=["Following two-step directions", "two-step directions"],
                verbatim_evidence=["completed 8 of 10"],
                evidence_source_session=1,
                substantively_treated=1,
                performance_observed=1,
                change_from_previous="new",
                carried_forward=0,
                newly_added=1,
            )
        ],
    )


def later_output(**changes: object) -> LaterSessionOutput:
    target = {
        "target_id": "T001",
        "verbatim_evidence": ["completed 9 of 10"],
        "evidence_source_session": 2,
        "substantively_treated": 1,
        "performance_observed": 1,
        "change_from_previous": "improved",
        "carried_forward": 0,
    }
    target.update(changes)
    return LaterSessionOutput.model_validate(
        {
            "client_id": "C001",
            "session_index": 2,
            "note_id": "N002",
            "existing_targets": [target],
            "new_targets": [],
        }
    )


def current_note() -> ClinicalNote:
    return ClinicalNote(
        "C001",
        2,
        "N002",
        "The client practiced two-step directions and completed 9 of 10.",
    )


def test_session1_keeps_treatment_and_observation_separate() -> None:
    value = discovered(treated=1, observed=0)
    assert (value.substantively_treated, value.performance_observed) == (1, 0)


def test_later_output_must_include_every_prior_target_once() -> None:
    output = later_output()
    validate_later_session(output, current_note(), previous_profile())
    output.existing_targets.append(output.existing_targets[0].model_copy())
    with pytest.raises(OutputValidationError, match="duplicate entries"):
        validate_later_session(output, current_note(), previous_profile())


def test_carry_forward_is_preserved_and_audited() -> None:
    output = later_output(
        verbatim_evidence=["completed 8 of 10"],
        evidence_source_session=1,
        substantively_treated=0,
        performance_observed=0,
        change_from_previous="not_assessed",
        carried_forward=1,
    )
    validate_later_session(output, current_note(), previous_profile())
    assert audit_later_session_quality(output, current_note(), previous_profile()) == []


def test_bad_carry_forward_is_flagged_but_not_rejected() -> None:
    output = later_output(
        verbatim_evidence=["changed old evidence"],
        evidence_source_session=2,
        change_from_previous="stable",
        carried_forward=1,
    )
    validate_later_session(output, current_note(), previous_profile())
    codes = {
        warning["code"]
        for warning in audit_later_session_quality(
            output,
            current_note(),
            previous_profile(),
        )
    }
    assert {
        "CARRY_FORWARD_EVIDENCE_CHANGED",
        "CARRY_FORWARD_SOURCE_CHANGED",
        "CARRY_FORWARD_WITH_CURRENT_ACTIVITY",
        "CARRY_FORWARD_WITH_CHANGE_JUDGMENT",
    } <= codes


def test_new_target_gets_next_python_assigned_id() -> None:
    registry = TargetRegistry(client_id="C001")
    registry.add_discovered(discovered())
    candidate = NewTargetCandidate(
        proposed_label="Answering where questions",
        definition="Answering questions about location.",
        aliases_in_note=["where questions"],
        verbatim_evidence=["answered where questions"],
        substantively_treated=1,
        performance_observed=1,
    )
    assert registry.add_new(candidate, 2) == "T002"
    assert registry.targets[-1].first_observed_session == 2


def test_later_prompt_contains_only_prior_json_and_current_raw_note() -> None:
    prompts = load_prompts()
    prompt = build_later_prompt(
        prompts["later_session"],
        previous_profile().model_dump(),
        ClinicalNote("C001", 2, "N002", "CURRENT RAW NOTE UNIQUE"),
    )
    assert "CURRENT RAW NOTE UNIQUE" in prompt
    assert '"session_index":1' in prompt
    assert "completed 8 of 10" in prompt
    assert "OLD RAW NOTE SECRET" not in prompt
    assert "NOTES_THROUGH_CURRENT_SESSION" not in prompt


def test_non_verbatim_evidence_is_preserved_and_flagged() -> None:
    note = ClinicalNote("C001", 1, "N001", "The client completed 8 of 10.")
    output = Session1Output(
        client_id="C001",
        session_index=1,
        note_id="N001",
        targets=[discovered()],
    )
    output.targets[0].verbatim_evidence = ["completed every trial"]
    validate_session1(output, note)
    assert {item["code"] for item in audit_session1_quality(output, note)} == {
        "NON_VERBATIM_EVIDENCE"
    }


def test_schema_rejects_boolean_in_binary_field() -> None:
    payload = json.loads(discovered().model_dump_json())
    payload["substantively_treated"] = True
    with pytest.raises(ValidationError):
        DiscoveredTarget.model_validate(payload)
