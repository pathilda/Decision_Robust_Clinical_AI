from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from clinical_target_extraction.src.data import ClinicalNote
from clinical_target_extraction.src.prompts import build_later_prompt, load_prompts
from clinical_target_extraction.src.registry import TargetRegistry
from clinical_target_extraction.src.schemas import (
    CanonicalizationOutput,
    DiscoveredTarget,
    LaterSessionOutput,
    NewTargetCandidate,
    PairwiseComparison,
    Session1Output,
)
from clinical_target_extraction.src.validate import (
    OutputValidationError,
    validate_later_session,
    validate_session1,
)


def discovered(*, treated: int, observed: int) -> DiscoveredTarget:
    return DiscoveredTarget(
        proposed_label="Following two-step directions",
        definition="Completing spoken directions containing two sequential steps.",
        aliases_in_note=["two-step directions"],
        substantively_treated=treated,
        performance_observed=observed,
        target_evidence=["two-step directions"],
        treatment_evidence=["practiced two-step directions"] if treated else [],
        performance_evidence=["completed 8 of 10"] if observed else [],
        rationale="The note directly documents the target.",
    )


def registry() -> TargetRegistry:
    value = TargetRegistry(client_id="C001")
    value.add_discovered(discovered(treated=1, observed=1))
    return value


def later_output(
    *,
    treated: int = 1,
    observed: int = 1,
    comparable: int = 1,
    better: int = 0,
    worse: int = 0,
) -> LaterSessionOutput:
    current_quote = "completed 9 of 10"
    prior_quote = "completed 8 of 10"
    return LaterSessionOutput.model_validate(
        {
            "client_id": "C001",
            "session_index": 2,
            "note_id": "N002",
            "target_assessments": [
                {
                    "target_id": "T001",
                    "substantively_treated": treated,
                    "performance_observed": observed,
                    "treatment_evidence": ["practiced two-step directions"] if treated else [],
                    "performance_evidence": [current_quote] if observed else [],
                    "comparisons": [
                        {
                            "prior_session_index": 1,
                            "comparable": comparable,
                            "better": better,
                            "worse": worse,
                            "current_evidence": [current_quote] if comparable else [],
                            "prior_evidence": [prior_quote] if comparable else [],
                            "comparison_basis": "Accuracy is compared under the same task.",
                        }
                    ],
                    "rationale": "Direct practice and an accuracy measure are documented.",
                }
            ],
            "new_target_candidates": [],
        }
    )


def validation_context() -> tuple[ClinicalNote, dict[int, ClinicalNote], TargetRegistry]:
    first = ClinicalNote(
        "C001",
        1,
        "N001",
        "The client practiced two-step directions and completed 8 of 10.",
    )
    second = ClinicalNote(
        "C001",
        2,
        "N002",
        "The client practiced two-step directions and completed 9 of 10.",
    )
    return second, {1: first, 2: second}, registry()


def validate_later(output: LaterSessionOutput, prior_observed: int = 1) -> None:
    current, notes, target_registry = validation_context()
    validate_later_session(
        output,
        current,
        notes,
        target_registry,
        {(1, "T001"): prior_observed},
    )


def test_target_treated_and_performance_observed() -> None:
    value = discovered(treated=1, observed=1)
    assert (value.substantively_treated, value.performance_observed) == (1, 1)


def test_target_treated_without_evaluable_performance() -> None:
    value = discovered(treated=1, observed=0)
    assert (value.substantively_treated, value.performance_observed) == (1, 0)


def test_performance_observed_without_treatment() -> None:
    value = discovered(treated=0, observed=1)
    assert (value.substantively_treated, value.performance_observed) == (0, 1)


def test_prior_session_without_observation_is_not_comparable() -> None:
    output = later_output(comparable=0)
    validate_later(output, prior_observed=0)

    with pytest.raises(OutputValidationError, match="comparable must be 0"):
        validate_later(later_output(comparable=1), prior_observed=0)


@pytest.mark.parametrize(
    ("better", "worse"),
    [(1, 0), (0, 1), (0, 0)],
    ids=["improves", "worsens", "no_meaningful_difference"],
)
def test_valid_comparable_directions(better: int, worse: int) -> None:
    validate_later(later_output(better=better, worse=worse))


def test_alias_candidate_does_not_create_new_id() -> None:
    target_registry = registry()
    candidate = NewTargetCandidate(
        proposed_label="two-step commands",
        definition="Completing two sequential spoken commands.",
        evidence=["two-step commands"],
        substantively_treated=1,
        performance_observed=1,
        possible_existing_target_id="T001",
        novelty_rationale="Potential wording variant.",
    )
    decision = CanonicalizationOutput(
        candidate_label="two-step commands",
        decision="same_as_existing",
        matched_target_id="T001",
        recommended_label="Following two-step directions",
        recommended_definition="Completing spoken directions containing two sequential steps.",
        rationale="The constructs are equivalent.",
    )
    assigned = target_registry.apply_canonicalization(candidate, decision, 2)
    assert assigned == "T001"
    assert len(target_registry.targets) == 1
    assert "two-step commands" in target_registry.targets[0].aliases


def test_genuinely_new_target_gets_next_stable_id() -> None:
    target_registry = registry()
    candidate = NewTargetCandidate(
        proposed_label="Answering where questions",
        definition="Answering questions about location.",
        evidence=["answered where questions"],
        substantively_treated=1,
        performance_observed=1,
        possible_existing_target_id=None,
        novelty_rationale="No equivalent target exists.",
    )
    decision = CanonicalizationOutput(
        candidate_label="Answering where questions",
        decision="genuinely_new",
        matched_target_id=None,
        recommended_label="Answering where questions",
        recommended_definition="Providing appropriate answers to questions about location.",
        rationale="This is a distinct construct.",
    )
    assigned = target_registry.apply_canonicalization(candidate, decision, 2)
    assert assigned == "T002"
    assert target_registry.targets[-1].first_observed_session == 2


def test_future_note_never_enters_earlier_prompt() -> None:
    prompts = load_prompts()
    notes = [
        ClinicalNote("C001", 1, "N001", "FIRST NOTE UNIQUE TEXT"),
        ClinicalNote("C001", 2, "N002", "SECOND NOTE UNIQUE TEXT"),
        ClinicalNote("C001", 3, "N003", "FUTURE SECRET TEXT"),
    ]
    prompt = build_later_prompt(
        prompts["later_session"],
        registry(),
        notes[:2],
        [{"session_index": 1, "target_records": []}],
    )
    assert "FIRST NOTE UNIQUE TEXT" in prompt
    assert "SECOND NOTE UNIQUE TEXT" in prompt
    assert "FUTURE SECRET TEXT" not in prompt
    assert '<SESSION index="1" note_id="N001">' in prompt
    assert '<SESSION index="2" note_id="N002">' in prompt


def test_non_verbatim_evidence_is_rejected() -> None:
    note = ClinicalNote(
        "C001",
        1,
        "N001",
        "The client practiced two-step directions and completed 8 of 10.",
    )
    output = Session1Output(
        client_id="C001",
        session_index=1,
        note_id="N001",
        targets=[discovered(treated=1, observed=1)],
    )
    output.targets[0].performance_evidence = ["completed all trials independently"]
    with pytest.raises(OutputValidationError, match="not a verbatim excerpt"):
        validate_session1(output, note)


def test_better_and_worse_cannot_both_equal_one() -> None:
    with pytest.raises(ValidationError, match="cannot both equal 1"):
        PairwiseComparison(
            prior_session_index=1,
            comparable=1,
            better=1,
            worse=1,
            current_evidence=[],
            prior_evidence=[],
            comparison_basis="Invalid direction encoding.",
        )


def test_noncomparable_direction_must_be_zeroed() -> None:
    with pytest.raises(ValidationError, match="non-comparable"):
        PairwiseComparison(
            prior_session_index=1,
            comparable=0,
            better=1,
            worse=0,
            current_evidence=[],
            prior_evidence=[],
            comparison_basis="Invalid non-comparable encoding.",
        )


def test_unknown_and_duplicate_target_ids_are_rejected() -> None:
    output = later_output()
    duplicate = output.target_assessments[0].model_copy(deep=True)
    output.target_assessments.append(duplicate)
    with pytest.raises(OutputValidationError, match="duplicate entries"):
        validate_later(output)


def test_schema_rejects_non_integer_binary_values() -> None:
    payload = json.loads(discovered(treated=1, observed=1).model_dump_json())
    payload["substantively_treated"] = True
    with pytest.raises(ValidationError):
        DiscoveredTarget.model_validate(payload)
