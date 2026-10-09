"""Strict structured-output schema for pre-treatment classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TreatmentCategory = Literal[
    "Easy to treatment",
    "Moderate to treatment",
    "severe to treatment",
]

TREATMENT_CATEGORIES: tuple[str, ...] = (
    "Easy to treatment",
    "Moderate to treatment",
    "severe to treatment",
)


class ClassificationOutput(BaseModel):
    """The only two values the LLM may return for a client."""

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1)
    treatment_category: TreatmentCategory


class ClassificationWithBriefReasoningOutput(BaseModel):
    """Classification plus a short, user-visible evidence summary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_id: str = Field(min_length=1)
    brief_reasoning: str = Field(min_length=1, max_length=1000)
    treatment_category: TreatmentCategory


ClassificationResultOutput = (
    ClassificationOutput | ClassificationWithBriefReasoningOutput
)
ModelOutput = ClassificationResultOutput


def schema_for_prompt_style(
    prompt_style: str,
) -> type[ClassificationOutput] | type[ClassificationWithBriefReasoningOutput]:
    if prompt_style == "standard":
        return ClassificationOutput
    if prompt_style == "brief-reasoning":
        return ClassificationWithBriefReasoningOutput
    raise ValueError(f"Unknown prompt style: {prompt_style!r}")
