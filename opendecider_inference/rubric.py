"""The shared three-category treatment-difficulty decision contract."""

from __future__ import annotations

import hashlib
import json


MODEL_LABEL = "opendecider"
QUESTION_NAME = "treatment_category"
QUESTION_INSTRUCTIONS = (
    "Based only on the supplied pre-treatment information, classify the client's "
    "expected treatment difficulty. Use both source sections when present. If the "
    "sources conflict, weigh the full clinical picture. Do not use information "
    "outside the note and do not infer absent facts. Treat the labels as relative "
    "treatment-complexity categories, not diagnoses, moral judgments, or predictions "
    "of a person's worth."
)
CATEGORY_CRITERIA: dict[str, str] = {
    "Easy to treatment": (
        "The documented needs appear relatively focused, lower intensity, and "
        "straightforward to address, with few evident complicating factors."
    ),
    "Moderate to treatment": (
        "The documented needs or barriers show meaningful complexity and are likely "
        "to require a moderate level of treatment intensity, coordination, or support."
    ),
    "severe to treatment": (
        "The documented needs, risks, functional impacts, or barriers appear highly "
        "complex or intensive and are likely to require substantial treatment effort, "
        "coordination, or support."
    ),
}
CATEGORIES = tuple(CATEGORY_CRITERIA)


def decision_questions() -> dict[str, dict[str, object]]:
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTIONS,
            "criteria": dict(CATEGORY_CRITERIA),
        }
    }


def rubric_hash() -> str:
    payload = {
        "question_name": QUESTION_NAME,
        "instructions": QUESTION_INSTRUCTIONS,
        "criteria": CATEGORY_CRITERIA,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
