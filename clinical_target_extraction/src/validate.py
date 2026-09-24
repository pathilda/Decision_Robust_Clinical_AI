"""Context-aware logical and verbatim-evidence validation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .data import ClinicalNote
from .registry import TargetRegistry
from .schemas import (
    CanonicalizationOutput,
    LaterSessionOutput,
    Session1Output,
)


class OutputValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _check_quotes(quotes: Iterable[str], note_text: str, location: str, errors: list[str]) -> None:
    normalized_note = normalize_whitespace(note_text)
    for quote in quotes:
        normalized_quote = normalize_whitespace(quote)
        if not normalized_quote:
            errors.append(f"{location}: evidence excerpts cannot be empty")
        elif normalized_quote not in normalized_note:
            errors.append(f"{location}: evidence is not a verbatim excerpt: {quote!r}")


def validate_session1(output: Session1Output, note: ClinicalNote) -> None:
    errors: list[str] = []
    if output.client_id != note.client_id:
        errors.append(f"client_id mismatch: expected {note.client_id!r}")
    if output.session_index != note.session_index:
        errors.append(f"session_index mismatch: expected {note.session_index}")
    if output.note_id != note.note_id:
        errors.append(f"note_id mismatch: expected {note.note_id!r}")
    labels: set[str] = set()
    for index, target in enumerate(output.targets):
        if not target.target_evidence:
            errors.append(f"targets[{index}].target_evidence: at least one excerpt is required")
        if target.substantively_treated == 1 and not target.treatment_evidence:
            errors.append(
                f"targets[{index}].treatment_evidence: required when substantively_treated=1"
            )
        if target.performance_observed == 1 and not target.performance_evidence:
            errors.append(
                f"targets[{index}].performance_evidence: required when performance_observed=1"
            )
        key = target.proposed_label.strip().casefold()
        if key in labels:
            errors.append(f"targets[{index}]: duplicate proposed label {target.proposed_label!r}")
        labels.add(key)
        _check_quotes(
            target.target_evidence,
            note.note_text,
            f"targets[{index}].target_evidence",
            errors,
        )
        _check_quotes(
            target.treatment_evidence,
            note.note_text,
            f"targets[{index}].treatment_evidence",
            errors,
        )
        _check_quotes(
            target.performance_evidence,
            note.note_text,
            f"targets[{index}].performance_evidence",
            errors,
        )
    if errors:
        raise OutputValidationError(errors)


def validate_later_session(
    output: LaterSessionOutput,
    current_note: ClinicalNote,
    notes_by_session: dict[int, ClinicalNote],
    registry: TargetRegistry,
    prior_observed: dict[tuple[int, str], int],
) -> None:
    errors: list[str] = []
    if output.client_id != current_note.client_id:
        errors.append(f"client_id mismatch: expected {current_note.client_id!r}")
    if output.session_index != current_note.session_index:
        errors.append(f"session_index mismatch: expected {current_note.session_index}")
    if output.note_id != current_note.note_id:
        errors.append(f"note_id mismatch: expected {current_note.note_id!r}")

    expected_targets = [target.target_id for target in registry.targets]
    actual_targets = [assessment.target_id for assessment in output.target_assessments]
    _check_exact_members(actual_targets, expected_targets, "target_assessments", errors)
    expected_sessions = list(range(1, current_note.session_index))

    for assessment in output.target_assessments:
        location = f"target_assessments[{assessment.target_id}]"
        if assessment.substantively_treated == 1 and not assessment.treatment_evidence:
            errors.append(f"{location}.treatment_evidence: required when substantively_treated=1")
        if assessment.performance_observed == 1 and not assessment.performance_evidence:
            errors.append(f"{location}.performance_evidence: required when performance_observed=1")
        _check_quotes(
            assessment.treatment_evidence,
            current_note.note_text,
            f"{location}.treatment_evidence",
            errors,
        )
        _check_quotes(
            assessment.performance_evidence,
            current_note.note_text,
            f"{location}.performance_evidence",
            errors,
        )
        actual_sessions = [comparison.prior_session_index for comparison in assessment.comparisons]
        _check_exact_members(actual_sessions, expected_sessions, f"{location}.comparisons", errors)
        for comparison in assessment.comparisons:
            prior_index = comparison.prior_session_index
            prior_note = notes_by_session.get(prior_index)
            if prior_note is None:
                errors.append(f"{location}: unknown prior session {prior_index}")
                continue
            comparison_location = f"{location}.comparisons[{prior_index}]"
            if comparison.comparable == 1:
                if not comparison.current_evidence:
                    errors.append(
                        f"{comparison_location}.current_evidence: required when comparable=1"
                    )
                if not comparison.prior_evidence:
                    errors.append(
                        f"{comparison_location}.prior_evidence: required when comparable=1"
                    )
            _check_quotes(
                comparison.current_evidence,
                current_note.note_text,
                f"{comparison_location}.current_evidence",
                errors,
            )
            _check_quotes(
                comparison.prior_evidence,
                prior_note.note_text,
                f"{comparison_location}.prior_evidence",
                errors,
            )
            prior_o = prior_observed.get((prior_index, assessment.target_id), 0)
            observation_missing = assessment.performance_observed == 0 or prior_o == 0
            if observation_missing and comparison.comparable != 0:
                errors.append(
                    f"{comparison_location}: comparable must be 0 when current or prior "
                    "performance_observed is 0"
                )

    registered_ids = set(expected_targets)
    for index, candidate in enumerate(output.new_target_candidates):
        if not candidate.evidence:
            errors.append(
                f"new_target_candidates[{index}].evidence: at least one excerpt is required"
            )
        _check_quotes(
            candidate.evidence,
            current_note.note_text,
            f"new_target_candidates[{index}]",
            errors,
        )
        if (
            candidate.possible_existing_target_id is not None
            and candidate.possible_existing_target_id not in registered_ids
        ):
            errors.append(
                f"new_target_candidates[{index}]: unknown possible_existing_target_id "
                f"{candidate.possible_existing_target_id!r}"
            )
    if errors:
        raise OutputValidationError(errors)


def validate_canonicalization(
    output: CanonicalizationOutput,
    candidate_label: str,
    registry: TargetRegistry,
) -> None:
    errors: list[str] = []
    if output.candidate_label != candidate_label:
        errors.append(f"candidate_label mismatch: expected {candidate_label!r}")
    if output.matched_target_id is not None:
        known = {target.target_id for target in registry.targets}
        if output.matched_target_id not in known:
            errors.append(f"unknown matched_target_id {output.matched_target_id!r}")
    if errors:
        raise OutputValidationError(errors)


def _check_exact_members(
    actual: list[int] | list[str],
    expected: list[int] | list[str],
    location: str,
    errors: list[str],
) -> None:
    if len(actual) != len(set(actual)):
        errors.append(f"{location}: duplicate entries: {actual}")
    if set(actual) != set(expected):
        errors.append(f"{location}: expected exactly {expected}, found {actual}")
