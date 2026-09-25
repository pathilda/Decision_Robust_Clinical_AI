"""Structural validation plus non-blocking evidence and carry-forward audits."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .data import ClinicalNote
from .schemas import LaterSessionOutput, Session1Output, SessionProfile


class OutputValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def validate_session1(output: Session1Output, note: ClinicalNote) -> None:
    errors = _metadata_errors(output.client_id, output.session_index, output.note_id, note)
    labels: set[str] = set()
    for index, target in enumerate(output.targets):
        key = normalize_whitespace(target.proposed_label).casefold()
        if key in labels:
            errors.append(f"targets[{index}]: duplicate proposed label {target.proposed_label!r}")
        labels.add(key)
    if errors:
        raise OutputValidationError(errors)


def validate_later_session(
    output: LaterSessionOutput,
    current_note: ClinicalNote,
    previous_profile: SessionProfile,
) -> None:
    """Reject only structural state-transition errors; clinical paradoxes are audited."""

    errors = _metadata_errors(
        output.client_id,
        output.session_index,
        output.note_id,
        current_note,
    )
    if previous_profile.client_id != current_note.client_id:
        errors.append("previous profile belongs to a different client")
    if previous_profile.session_index != current_note.session_index - 1:
        errors.append(
            "previous profile must be from the immediately previous session: "
            f"expected {current_note.session_index - 1}, "
            f"found {previous_profile.session_index}"
        )

    expected_ids = [target.target_id for target in previous_profile.targets]
    actual_ids = [target.target_id for target in output.existing_targets]
    _check_exact_members(actual_ids, expected_ids, "existing_targets", errors)

    new_labels: set[str] = set()
    for index, target in enumerate(output.new_targets):
        key = normalize_whitespace(target.proposed_label).casefold()
        if key in new_labels:
            errors.append(
                f"new_targets[{index}]: duplicate proposed label {target.proposed_label!r}"
            )
        new_labels.add(key)

    if errors:
        raise OutputValidationError(errors)


def audit_session1_quality(
    output: Session1Output,
    note: ClinicalNote,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for index, target in enumerate(output.targets):
        location = f"targets[{index}]"
        if target.substantively_treated and not target.verbatim_evidence:
            _warning(warnings, "TREATED_WITHOUT_EVIDENCE", location)
        if target.performance_observed and not target.verbatim_evidence:
            _warning(warnings, "OBSERVED_WITHOUT_EVIDENCE", location)
        _audit_evidence(
            target.verbatim_evidence,
            note.note_text,
            f"{location}.verbatim_evidence",
            warnings,
        )
    return warnings


def audit_later_session_quality(
    output: LaterSessionOutput,
    current_note: ClinicalNote,
    previous_profile: SessionProfile,
) -> list[dict[str, str]]:
    """Keep imperfect evidence or logical combinations, but make them auditable."""

    warnings: list[dict[str, str]] = []
    previous_by_id = {target.target_id: target for target in previous_profile.targets}

    for target in output.existing_targets:
        location = f"existing_targets[{target.target_id}]"
        previous = previous_by_id.get(target.target_id)
        if previous is None:
            continue
        if target.carried_forward:
            if target.verbatim_evidence != previous.verbatim_evidence:
                _warning(warnings, "CARRY_FORWARD_EVIDENCE_CHANGED", location)
            if target.evidence_source_session != previous.evidence_source_session:
                _warning(warnings, "CARRY_FORWARD_SOURCE_CHANGED", location)
            if target.substantively_treated or target.performance_observed:
                _warning(warnings, "CARRY_FORWARD_WITH_CURRENT_ACTIVITY", location)
            if target.change_from_previous != "not_assessed":
                _warning(warnings, "CARRY_FORWARD_WITH_CHANGE_JUDGMENT", location)
        else:
            _audit_evidence(
                target.verbatim_evidence,
                current_note.note_text,
                f"{location}.verbatim_evidence",
                warnings,
            )
            if target.evidence_source_session != current_note.session_index:
                _warning(warnings, "CURRENT_EVIDENCE_WRONG_SOURCE_SESSION", location)
            if not target.substantively_treated and not target.performance_observed:
                _warning(warnings, "CURRENT_ENTRY_WITHOUT_ACTIVITY", location)
            if (
                target.change_from_previous in {"improved", "stable", "worsened"}
                and not target.performance_observed
            ):
                _warning(warnings, "CHANGE_WITHOUT_OBSERVED_PERFORMANCE", location)

    for index, target in enumerate(output.new_targets):
        location = f"new_targets[{index}]"
        _audit_evidence(
            target.verbatim_evidence,
            current_note.note_text,
            f"{location}.verbatim_evidence",
            warnings,
        )
        if target.substantively_treated and not target.verbatim_evidence:
            _warning(warnings, "TREATED_WITHOUT_EVIDENCE", location)
        if target.performance_observed and not target.verbatim_evidence:
            _warning(warnings, "OBSERVED_WITHOUT_EVIDENCE", location)
    return warnings


def _metadata_errors(
    client_id: str,
    session_index: int,
    note_id: str,
    note: ClinicalNote,
) -> list[str]:
    errors: list[str] = []
    if client_id != note.client_id:
        errors.append(f"client_id mismatch: expected {note.client_id!r}")
    if session_index != note.session_index:
        errors.append(f"session_index mismatch: expected {note.session_index}")
    if note_id != note.note_id:
        errors.append(f"note_id mismatch: expected {note.note_id!r}")
    return errors


def _audit_evidence(
    values: Iterable[str],
    note_text: str,
    location: str,
    warnings: list[dict[str, str]],
) -> None:
    normalized_note = normalize_whitespace(note_text)
    for value in values:
        if normalize_whitespace(value) not in normalized_note:
            _warning(warnings, "NON_VERBATIM_EVIDENCE", location)


def _warning(
    warnings: list[dict[str, str]],
    code: str,
    location: str,
) -> None:
    warnings.append({"code": code, "location": location})


def _check_exact_members(
    actual: list[str],
    expected: list[str],
    location: str,
    errors: list[str],
) -> None:
    if len(actual) != len(set(actual)):
        errors.append(f"{location}: duplicate entries: {actual}")
    if set(actual) != set(expected):
        errors.append(f"{location}: expected exactly {expected}, found {actual}")
