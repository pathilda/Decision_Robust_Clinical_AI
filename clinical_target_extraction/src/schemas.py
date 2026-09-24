"""Strict Pydantic schemas for every model-produced response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class StrictModel(BaseModel):
    """Forbid model output fields that are not part of the declared contract."""

    model_config = ConfigDict(extra="forbid")


Binary = StrictInt


def _binary_field() -> int:
    return Field(ge=0, le=1)


class DiscoveredTarget(StrictModel):
    proposed_label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    aliases_in_note: list[str]
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()
    target_evidence: list[str]
    treatment_evidence: list[str]
    performance_evidence: list[str]
    rationale: str = Field(min_length=1)


class Session1Output(StrictModel):
    client_id: str = Field(min_length=1)
    session_index: Literal[1]
    note_id: str = Field(min_length=1)
    targets: list[DiscoveredTarget]


class PairwiseComparison(StrictModel):
    prior_session_index: int = Field(strict=True, ge=1)
    comparable: Binary = _binary_field()
    better: Binary = _binary_field()
    worse: Binary = _binary_field()
    current_evidence: list[str]
    prior_evidence: list[str]
    comparison_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_direction(self) -> "PairwiseComparison":
        if self.better + self.worse > 1:
            raise ValueError("better and worse cannot both equal 1")
        if self.comparable == 0 and (self.better != 0 or self.worse != 0):
            raise ValueError("non-comparable records must set better=0 and worse=0")
        return self


class TargetAssessment(StrictModel):
    target_id: str = Field(pattern=r"^T\d{3,}$")
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()
    treatment_evidence: list[str]
    performance_evidence: list[str]
    comparisons: list[PairwiseComparison]
    rationale: str = Field(min_length=1)


class NewTargetCandidate(StrictModel):
    proposed_label: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    evidence: list[str]
    substantively_treated: Binary = _binary_field()
    performance_observed: Binary = _binary_field()
    possible_existing_target_id: str | None
    novelty_rationale: str = Field(min_length=1)


class LaterSessionOutput(StrictModel):
    client_id: str = Field(min_length=1)
    session_index: int = Field(strict=True, ge=2)
    note_id: str = Field(min_length=1)
    target_assessments: list[TargetAssessment]
    new_target_candidates: list[NewTargetCandidate]


class CanonicalizationOutput(StrictModel):
    candidate_label: str = Field(min_length=1)
    decision: Literal["same_as_existing", "genuinely_new"]
    matched_target_id: str | None
    recommended_label: str = Field(min_length=1)
    recommended_definition: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_match(self) -> "CanonicalizationOutput":
        if self.decision == "same_as_existing" and self.matched_target_id is None:
            raise ValueError("matched_target_id is required for same_as_existing")
        if self.decision == "genuinely_new" and self.matched_target_id is not None:
            raise ValueError("matched_target_id must be null for genuinely_new")
        return self


ModelOutput = Session1Output | LaterSessionOutput | CanonicalizationOutput
