"""Strict structured-output schema for pre-treatment classification."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


PROBABILITY_COLUMNS = dict(
    zip(TREATMENT_CATEGORIES, ("probability_easy", "probability_moderate", "probability_severe"))
)
PROBABILITY_STYLES = {"probabilities", "brief-reasoning-probabilities"}
PROBABILITY_TOLERANCE = 1e-6
PROBABILITY_CONTRACT = {
    "version": "llm_probabilities_v1",
    "probability_source": "llm_self_reported",
    "probabilities_calibrated": False,
    "sum_absolute_tolerance": PROBABILITY_TOLERANCE,
    "confidence_absolute_tolerance": PROBABILITY_TOLERANCE,
    "tie_order": list(TREATMENT_CATEGORIES),
    "normalization": "none; invalid responses get one model repair attempt",
    "json_order": "confidence first; treatment_category last; schema field order enforced",
}
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class CategoryProbabilities(BaseModel):
    """Exactly the existing three category labels, as numeric probabilities."""

    model_config = ConfigDict(extra="forbid")

    easy: Probability = Field(alias=TREATMENT_CATEGORIES[0])
    moderate: Probability = Field(alias=TREATMENT_CATEGORIES[1])
    severe: Probability = Field(alias=TREATMENT_CATEGORIES[2])


class ProbabilityOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: Probability
    probabilities: CategoryProbabilities
    client_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_probabilities(self):
        values = self.probabilities.model_dump(by_alias=True)
        # Compare reported decimal values so a boundary such as 3 * 0.333333 is accepted.
        tolerance = Decimal(str(PROBABILITY_TOLERANCE))
        total = sum((Decimal(str(value)) for value in values.values()), Decimal(0))
        if abs(total - 1) > tolerance:
            raise ValueError(f"Class probabilities must sum to 1 (tolerance 1e-6); got {total}")
        # Python max keeps the first category in the declared order for exact ties.
        chosen = max(TREATMENT_CATEGORIES, key=values.__getitem__)
        if self.treatment_category != chosen:
            raise ValueError(
                f"treatment_category must be {chosen!r}: select the largest probability; "
                "break exact ties in Easy, Moderate, severe order"
            )
        if abs(Decimal(str(self.confidence)) - Decimal(str(values[chosen]))) > tolerance:
            raise ValueError("confidence must equal the selected category's probability")
        return self


class ClassificationWithProbabilitiesOutput(ProbabilityOutputBase):
    treatment_category: TreatmentCategory


class ClassificationWithBriefReasoningProbabilitiesOutput(ProbabilityOutputBase):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    brief_reasoning: str = Field(min_length=1, max_length=1000)
    treatment_category: TreatmentCategory


ClassificationResultOutput = (
    ClassificationOutput
    | ClassificationWithBriefReasoningOutput
    | ClassificationWithProbabilitiesOutput
    | ClassificationWithBriefReasoningProbabilitiesOutput
)
ModelOutput = ClassificationResultOutput


def schema_for_prompt_style(
    prompt_style: str,
) -> type[ClassificationResultOutput]:
    if prompt_style == "standard":
        return ClassificationOutput
    if prompt_style == "brief-reasoning":
        return ClassificationWithBriefReasoningOutput
    if prompt_style == "probabilities":
        return ClassificationWithProbabilitiesOutput
    if prompt_style == "brief-reasoning-probabilities":
        return ClassificationWithBriefReasoningProbabilitiesOutput
    raise ValueError(f"Unknown prompt style: {prompt_style!r}")


def parse_classification_output(raw: str, schema: type[BaseModel]) -> BaseModel:
    """Enforce the requested generated JSON order without rewriting the model response."""
    if not issubclass(schema, ProbabilityOutputBase):
        return schema.model_validate_json(raw)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    parsed = json.loads(raw, object_pairs_hook=unique_object)
    expected = list(schema.model_fields)
    if not isinstance(parsed, dict) or list(parsed) != expected:
        raise ValueError(f"Return JSON fields in this exact order: {expected}")
    return schema.model_validate(parsed)
