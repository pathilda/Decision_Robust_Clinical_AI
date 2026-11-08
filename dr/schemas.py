"""JSON schemas for guided decoding (spec 4.3, 5.4, 6.2, 7.2) and a strict validator.

Static schemas are written to schemas/*.json with ``python -m dr.schemas`` and loaded from
there at run time; the allocation and tie-break schemas are built per request because their
enums are the actual client, clinician and candidate IDs. Property order is significant:
reasoning comes first and the decision last, and the ICF summary comes last.
"""

from __future__ import annotations

import argparse
import json
from functools import cache

from .common import SCHEMAS_DIR, classes_config

STATUS = ["known", "queried", "no_concern", "not_documented"]
GENERIC_QUALIFIERS = [0, 1, 2, 3, 4, 8, 9]
ENV_MAGNITUDES = [0, 1, 2, 3, 4, 8]
DIRECTIONS = ["barrier", "facilitator", "neither"]
YES_NO_ND = ["yes", "no", "not_documented"]
SETTINGS = ["home", "childcare_school", "community"]
PROFESSIONAL_STATUS = ["involved", "waitlisted", "referred", "queried"]
BODY_FUNCTIONS = [
    "articulation_phonology",
    "motor_speech_planning",
    "fluency",
    "voice_resonance",
    "receptive_language",
    "expressive_language",
    "social_communication_pragmatics",
    "hearing_auditory_processing",
    "oral_motor_feeding_swallowing",
]
ENVIRONMENTAL_FIELDS = [
    "caregiver_capacity",
    "home_community_environment",
    "service_access_social_determinants",
]
ARRAY_LIMITS = {"findings": 5, "other_professionals": 10, "developmental_factors": 5, "other": 5}


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def string():
    return {"type": "string"}


def enum(values, kind="string"):
    return {"type": kind, "enum": list(values)}


def array(items, max_items=None, min_items=None):
    value = {"type": "array", "items": items}
    if min_items is not None:
        value["minItems"] = min_items
    if max_items is not None:
        value["maxItems"] = max_items
    return value


def nullable_string():
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def qualified_field():
    return obj(
        {
            "status": enum(STATUS),
            "qualifier": enum(GENERIC_QUALIFIERS, "integer"),
            "description": string(),
            "evidence": string(),
        }
    )


def environmental_field():
    return obj(
        {
            "status": enum(STATUS),
            "direction": enum(DIRECTIONS),
            "magnitude": enum(ENV_MAGNITUDES, "integer"),
            "description": string(),
            "evidence": string(),
        }
    )


def described_field():
    return obj({"status": enum(STATUS), "description": string(), "evidence": string()})


def icf_schema():
    known_or_queried = enum(["known", "queried"])
    return obj(
        {
            "body_structures": obj(
                {
                    "status": enum(STATUS),
                    "findings": array(
                        obj(
                            {
                                "description": string(),
                                "status": known_or_queried,
                                "evidence": string(),
                            }
                        ),
                        ARRAY_LIMITS["findings"],
                    ),
                }
            ),
            "body_functions": obj({name: qualified_field() for name in BODY_FUNCTIONS}),
            "activity": qualified_field(),
            "participation": obj(
                {
                    "status": enum(STATUS),
                    "qualifier": enum(GENERIC_QUALIFIERS, "integer"),
                    "settings_affected": array(enum(SETTINGS), len(SETTINGS)),
                    "description": string(),
                    "evidence": string(),
                }
            ),
            "environmental_factors": obj(
                {
                    **{name: environmental_field() for name in ENVIRONMENTAL_FIELDS},
                    "family_physician_or_pediatrician_involved": enum(YES_NO_ND),
                    "other_professionals": array(
                        obj(
                            {
                                "role": string(),
                                "status": enum(PROFESSIONAL_STATUS),
                                "evidence": string(),
                            }
                        ),
                        ARRAY_LIMITS["other_professionals"],
                    ),
                }
            ),
            "personal_factors": obj(
                {
                    "age": string(),
                    "developmental_factors": array(
                        obj(
                            {
                                "description": string(),
                                "status": known_or_queried,
                                "evidence": string(),
                            }
                        ),
                        ARRAY_LIMITS["developmental_factors"],
                    ),
                    "behavioural_profile": described_field(),
                    "birth_history": described_field(),
                    "other": array(
                        obj({"description": string(), "evidence": string()}),
                        ARRAY_LIMITS["other"],
                    ),
                }
            ),
            "summary": obj(
                {
                    "areas_of_need": array(enum(BODY_FUNCTIONS), len(BODY_FUNCTIONS)),
                    "n_areas_of_need": {"type": "integer", "minimum": 0},
                    "max_severity_qualifier": enum(ENV_MAGNITUDES, "integer"),
                    "n_other_professionals": {"type": "integer", "minimum": 0},
                    "developmental_factors_present": enum(["none", "queried", "known"]),
                    "activity_or_participation_limited": enum(YES_NO_ND),
                    "caregiver_capacity_impacted": enum(YES_NO_ND),
                    "environmental_or_personal_factors_impacting_care": enum(YES_NO_ND),
                    "documentation_gaps": array(string(), 20),
                }
            ),
        }
    )


def label_schema():
    return obj(
        {
            "reasoning": string(),
            "fit_issue": nullable_string(),
            "class": enum(classes_config()["ids"], "integer"),
        }
    )


def distribution_schema():
    ids = [str(i) for i in classes_config()["ids"]]
    return obj(
        {
            "reasoning": string(),
            "fit_issue": nullable_string(),
            "probabilities": obj(
                {i: {"type": "number", "minimum": 0, "maximum": 1} for i in ids}
            ),
        }
    )


def score_schema():
    c = classes_config()
    return obj(
        {
            "reasoning": string(),
            "fit_issue": nullable_string(),
            "complexity_score": {"type": "number", "minimum": c["min"], "maximum": c["max"]},
        }
    )


def allocation_schema(client_ids, clinician_ids):
    n = len(client_ids)
    return obj(
        {
            "reasoning": string(),
            "assignment": array(
                obj(
                    {
                        "client_id": enum([str(c) for c in client_ids]),
                        "clinician_id": enum([str(c) for c in clinician_ids]),
                    }
                ),
                max_items=n,
                min_items=n,
            ),
        }
    )


def tiebreak_schema(k):
    return obj({"reasoning": string(), "chosen_candidate": enum(range(1, k + 1), "integer")})


STATIC = {
    "icf": icf_schema,
    "label": label_schema,
    "distribution": distribution_schema,
    "score": score_schema,
}


@cache
def load(name):
    """Static schema from schemas/<name>.json (must match the builder; see tests)."""
    return json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- validator
def validate(value, schema, path="$"):
    """Strict validator for the subset of JSON Schema used here. Returns error strings."""
    errors = []
    if "anyOf" in schema:
        if all(validate(value, option, path) for option in schema["anyOf"]):
            errors.append(f"{path}: matches no allowed alternative")
        return errors
    kind = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer()),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "null": lambda v: v is None,
        "boolean": lambda v: isinstance(v, bool),
    }
    if kind and not checks[kind](value):
        return [f"{path}: expected {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if kind == "number" or kind == "integer":
        if value != value or value in (float("inf"), float("-inf")):
            errors.append(f"{path}: non-finite number")
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if kind == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected {name!r}")
        for name, sub in properties.items():
            if name in value:
                errors.extend(validate(value[name], sub, f"{path}.{name}"))
    if kind == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for index, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the static JSON schemas to schemas/")
    parser.add_argument("--check", action="store_true", help="Only verify the files are current")
    args = parser.parse_args(argv)
    stale = []
    for name, builder in STATIC.items():
        path = SCHEMAS_DIR / f"{name}.json"
        text = json.dumps(builder(), indent=2) + "\n"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    if stale:
        print(f"stale schemas: {stale}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
