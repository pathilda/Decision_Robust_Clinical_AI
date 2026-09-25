"""Strict schemas for rolling clinical-target profile generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class StrictModel(BaseModel):
    """Forbid model output fields outside the declared JSON contract."""

    model_config = ConfigDict(extra="forbid")


Binary = StrictInt
ChangeLabel = Literal["improved", "stable", "worsened", "not_assessed"]
ProfileChangeLabel = Literal[
    "new", "improved", "stable", "worsened", "not_assessed"
]


def _binary_field() -> int:
    return Field(ge=0, le=1)


class DiscoveredTarget(StrictModel):
    """A target first found in session 1."""

    proposed_label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases_in_note: list[str]
    verbatim_evidence: list[str]
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()


class Session1Output(StrictModel):
    client_id: str = Field(min_length=1)
    session_index: Literal[1]
    note_id: str = Field(min_length=1)
    targets: list[DiscoveredTarget]


class ExistingTargetUpdate(StrictModel):
    """The current snapshot of one target already present in the prior profile."""

    target_id: str = Field(pattern=r"^T\d{3,}$")
    verbatim_evidence: list[str]
    evidence_source_session: int = Field(strict=True, ge=1)
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()
    change_from_previous: ChangeLabel
    carried_forward: Binary = _binary_field()


class NewTargetCandidate(StrictModel):
    """A genuinely new target found in the current note."""

    proposed_label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases_in_note: list[str]
    verbatim_evidence: list[str]
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()


class LaterSessionOutput(StrictModel):
    client_id: str = Field(min_length=1)
    session_index: int = Field(strict=True, ge=2)
    note_id: str = Field(min_length=1)
    existing_targets: list[ExistingTargetUpdate]
    new_targets: list[NewTargetCandidate]


class ProfileTarget(StrictModel):
    """One durable target entry passed to the next session."""

    target_id: str = Field(pattern=r"^T\d{3,}$")
    canonical_label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases: list[str]
    verbatim_evidence: list[str]
    evidence_source_session: int = Field(strict=True, ge=1)
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()
    change_from_previous: ProfileChangeLabel
    carried_forward: Binary = _binary_field()
    newly_added: Binary = _binary_field()


class SessionProfile(StrictModel):
    """Complete compact state handed from one session to the next."""

    client_id: str = Field(min_length=1)
    session_index: int = Field(strict=True, ge=1)
    note_id: str = Field(min_length=1)
    targets: list[ProfileTarget]


ModelOutput = Session1Output | LaterSessionOutput
