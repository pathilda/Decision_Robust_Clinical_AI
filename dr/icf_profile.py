"""ICF summary recomputation, consistency flags and the compact rendering (spec 4.3, 4.4).

``render_icf`` is the only way a client is described in the second and third inferences, so
rungs differ only in what is added to the prompt, never in how a client is described.
"""

from __future__ import annotations

import re

from .schemas import BODY_FUNCTIONS

SEVERITY_WORDS = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "profound",
    8: "severity not specified",
    9: "not applicable",
}
MAGNITUDE_WORDS = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "complete",
    8: "extent not specified",
}
DIRECTION_WORDS = {
    "barrier": "barrier",
    "facilitator": "facilitator",
    "neither": "neither barrier nor facilitator",
}
BODY_FUNCTION_LABELS = {
    "articulation_phonology": "Articulation/phonology",
    "motor_speech_planning": "Motor speech planning",
    "fluency": "Fluency",
    "voice_resonance": "Voice/resonance",
    "receptive_language": "Receptive language",
    "expressive_language": "Expressive language",
    "social_communication_pragmatics": "Social communication/pragmatics",
    "hearing_auditory_processing": "Hearing/auditory processing",
    "oral_motor_feeding_swallowing": "Oral-motor/feeding/swallowing",
}
SETTING_WORDS = {"home": "home", "childcare_school": "childcare/school", "community": "community"}
DOCUMENTED = ("known", "queried")
NEED_QUALIFIERS = (1, 2, 3, 4, 8)


def _text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _status(value):
    return str(value).replace("_", " ")


def _q(value):
    return int(value)


# ----------------------------------------------------------------------------- summary
def recompute_summary(profile):
    """Recompute every summary value that is derivable from the fields.

    Rules (documented in NOTES_existing_code.md):
    - areas_of_need: body functions with status known/queried and qualifier 1, 2, 3, 4 or 8.
    - max_severity_qualifier: highest of 1-4 among the areas of need; 8 if the areas carry
      only unspecified severities; 0 if there are no areas of need.
    - developmental_factors_present: known > queried > none over the listed factors.
    - Yes/no/not_documented fields: "yes" if any contributing field documents an impact;
      otherwise "no" if any contributing field is documented (known, queried or no_concern);
      otherwise "not_documented". Silence is never turned into "no".
    - documentation_gaps is a judgment ("would normally matter") and is kept from the model.
    """
    functions = profile["body_functions"]
    areas = [
        name
        for name in BODY_FUNCTIONS
        if functions[name]["status"] in DOCUMENTED
        and _q(functions[name]["qualifier"]) in NEED_QUALIFIERS
    ]
    qualifiers = [_q(functions[name]["qualifier"]) for name in areas]
    numeric = [q for q in qualifiers if q in (1, 2, 3, 4)]
    max_severity = max(numeric) if numeric else (8 if qualifiers else 0)

    statuses = [f["status"] for f in profile["personal_factors"]["developmental_factors"]]
    developmental = "known" if "known" in statuses else "queried" if "queried" in statuses else "none"

    def limited(field):
        return field["status"] in DOCUMENTED and _q(field["qualifier"]) in NEED_QUALIFIERS

    def tri(impacts, fields):
        if any(impacts):
            return "yes"
        if any(f["status"] != "not_documented" for f in fields):
            return "no"
        return "not_documented"

    activity, participation = profile["activity"], profile["participation"]
    env = profile["environmental_factors"]
    caregiver = env["caregiver_capacity"]

    def barrier(field):
        return (
            field["status"] in DOCUMENTED
            and field["direction"] == "barrier"
            and _q(field["magnitude"]) != 0
        )

    behaviour = profile["personal_factors"]["behavioural_profile"]
    contextual = [env["home_community_environment"], env["service_access_social_determinants"]]
    return {
        "areas_of_need": areas,
        "n_areas_of_need": len(areas),
        "max_severity_qualifier": max_severity,
        "n_other_professionals": len(env["other_professionals"]),
        "developmental_factors_present": developmental,
        "activity_or_participation_limited": tri(
            [limited(activity), limited(participation)], [activity, participation]
        ),
        "caregiver_capacity_impacted": tri([barrier(caregiver)], [caregiver]),
        "environmental_or_personal_factors_impacting_care": tri(
            [*(barrier(f) for f in contextual), behaviour["status"] in DOCUMENTED],
            [*contextual, behaviour],
        ),
        "documentation_gaps": list(profile["summary"].get("documentation_gaps", [])),
    }


def consistency_flags(profile):
    """Non-blocking flags for departures from the ICF prompt rules (stored with the record)."""
    flags = []

    def check(name, field, qualifier_key="qualifier"):
        status = field["status"]
        q = _q(field[qualifier_key]) if qualifier_key in field else None
        if status == "not_documented" and (_text(field.get("description")) or _text(field.get("evidence"))):
            flags.append(f"not_documented_with_text:{name}")
        if status in DOCUMENTED and not _text(field.get("evidence")):
            flags.append(f"missing_evidence:{name}")
        if q is not None and qualifier_key == "qualifier":
            if q == 0 and status != "no_concern":
                flags.append(f"qualifier0_without_no_concern:{name}")
            if status == "no_concern" and q not in (0, 9):
                flags.append(f"no_concern_with_problem_qualifier:{name}")

    for name in BODY_FUNCTIONS:
        check(name, profile["body_functions"][name])
    check("activity", profile["activity"])
    check("participation", profile["participation"])
    for name in ("caregiver_capacity", "home_community_environment", "service_access_social_determinants"):
        check(name, profile["environmental_factors"][name], qualifier_key="magnitude")
    if len({*profile["summary"]["areas_of_need"]}) != len(profile["summary"]["areas_of_need"]):
        flags.append("duplicate_areas_of_need")
    return flags


def summary_differences(model_summary, recomputed):
    keys = [k for k in recomputed if k != "documentation_gaps"]
    diff = []
    for key in keys:
        left, right = model_summary.get(key), recomputed[key]
        if key == "areas_of_need":
            left, right = sorted(set(left or [])), sorted(set(right))
        if left != right:
            diff.append(key)
    return diff


# ----------------------------------------------------------------------------- rendering
def _qualified(field):
    if field["status"] == "not_documented":
        return "not documented"
    q = _q(field["qualifier"])
    head = f"{SEVERITY_WORDS[q]} ({q}, {_status(field['status'])})."
    description = _text(field["description"])
    return f"{head} {description}".rstrip()


def _participation(field):
    if field["status"] == "not_documented":
        return "not documented"
    q = _q(field["qualifier"])
    settings = ", ".join(SETTING_WORDS[s] for s in dict.fromkeys(field["settings_affected"]))
    head = f"{SEVERITY_WORDS[q]} ({q}, {_status(field['status'])}), settings: {settings or 'none listed'}."
    return f"{head} {_text(field['description'])}".rstrip()


def _environmental(field):
    if field["status"] == "not_documented":
        return "not documented"
    m = _q(field["magnitude"])
    head = (
        f"{DIRECTION_WORDS[field['direction']]}, {MAGNITUDE_WORDS[m]} "
        f"({m}, {_status(field['status'])})."
    )
    return f"{head} {_text(field['description'])}".rstrip()


def _described(field):
    if field["status"] == "not_documented":
        return "not documented"
    return f"{_status(field['status'])}. {_text(field['description'])}".rstrip()


def _listing(items, fmt):
    rendered = [fmt(item) for item in items]
    rendered = [r for r in rendered if r]
    return "; ".join(rendered) if rendered else "none documented"


def render_icf(client_id, profile, summary):
    """Deterministic compact profile. ``summary`` must be the recomputed summary."""
    functions = profile["body_functions"]
    structures = profile["body_structures"]
    env = profile["environmental_factors"]
    personal = profile["personal_factors"]

    if structures["findings"]:
        body_structures = _listing(
            structures["findings"],
            lambda f: f"{_text(f['description'])} ({_status(f['status'])})",
        )
    elif structures["status"] == "no_concern":
        body_structures = "no concern"
    else:
        body_structures = "none documented"

    q = summary["max_severity_qualifier"]
    areas = ", ".join(BODY_FUNCTION_LABELS[a].lower() for a in summary["areas_of_need"])
    lines = [
        f"[Client {client_id}]",
        f"Age: {_text(personal['age']) or 'not documented'}",
        "Body functions:",
        *(f"- {BODY_FUNCTION_LABELS[n]}: {_qualified(functions[n])}" for n in BODY_FUNCTIONS),
        f"Body structures: {body_structures}",
        f"Activity: {_qualified(profile['activity'])}",
        f"Participation: {_participation(profile['participation'])}",
        f"Caregiver capacity: {_environmental(env['caregiver_capacity'])}",
        f"Home/community environment: {_environmental(env['home_community_environment'])}",
        "Service access/social determinants: "
        + _environmental(env["service_access_social_determinants"]),
        "Other professionals (beyond family physician/pediatrician): "
        + _listing(env["other_professionals"], lambda p: f"{_text(p['role'])} ({p['status']})"),
        "Developmental factors: "
        + _listing(
            personal["developmental_factors"],
            lambda f: f"{_text(f['description'])} ({_status(f['status'])})",
        ),
        f"Behavioural profile: {_described(personal['behavioural_profile'])}",
        f"Birth history: {_described(personal['birth_history'])}",
        "Other personal factors: " + _listing(personal["other"], lambda f: _text(f["description"])),
        (
            f"Summary: {summary['n_areas_of_need']} area(s) of need [{areas or 'none'}], "
            f"highest severity {SEVERITY_WORDS[q]}, "
            f"{summary['n_other_professionals']} other professional(s), "
            f"developmental factors {summary['developmental_factors_present']}, "
            "activity/participation limited "
            f"{_status(summary['activity_or_participation_limited'])}, "
            f"caregiver capacity impacted {_status(summary['caregiver_capacity_impacted'])}, "
            "environmental/personal factors impacting care "
            f"{_status(summary['environmental_or_personal_factors_impacting_care'])}."
        ),
    ]
    return "\n".join(lines)
