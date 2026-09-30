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


ModelOutput = ClassificationOutput
