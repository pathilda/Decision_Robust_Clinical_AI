"""Entirely SYNTHETIC ICF profiles for tests and structural prompt previews (no real data)."""

from __future__ import annotations

import random

from .schemas import BODY_FUNCTIONS


def _field(rng, documented_rate=0.5):
    roll = rng.random()
    if roll > documented_rate:
        return {"status": "not_documented", "qualifier": 9, "description": "", "evidence": ""}
    if roll < 0.08:
        return {"status": "no_concern", "qualifier": 0, "description": "Within normal limits.",
                "evidence": "WNL per assessment"}
    q = rng.choice([1, 2, 2, 3, 8])
    status = rng.choice(["known", "known", "queried"])
    return {"status": status, "qualifier": q,
            "description": f"Synthetic {status} difficulty, qualifier {q}.",
            "evidence": "synthetic quote"}


def _env(rng):
    status = rng.choice(["known", "not_documented", "no_concern", "queried"])
    if status == "not_documented":
        return {"status": status, "direction": "neither", "magnitude": 0, "description": "",
                "evidence": ""}
    if status == "no_concern":
        return {"status": status, "direction": "facilitator", "magnitude": 0,
                "description": "Supportive.", "evidence": "synthetic quote"}
    return {"status": status, "direction": rng.choice(["barrier", "facilitator"]),
            "magnitude": rng.choice([1, 2, 8]), "description": "Synthetic environment note.",
            "evidence": "synthetic quote"}


def synthetic_profile(seed):
    rng = random.Random(seed)
    functions = {name: _field(rng, 0.45) for name in BODY_FUNCTIONS}
    professionals = [
        {"role": role, "status": rng.choice(["involved", "waitlisted", "referred", "queried"]),
         "evidence": "synthetic quote"}
        for role in rng.sample(["OT", "PT", "Audiology", "Social work", "FAST team"], rng.randint(0, 3))
    ]
    developmental = [
        {"description": d, "status": rng.choice(["known", "queried"]), "evidence": "synthetic quote"}
        for d in rng.sample(["Global developmental delay", "Autism", "Motor delay"], rng.randint(0, 2))
    ]
    profile = {
        "body_structures": {"status": "not_documented", "findings": []},
        "body_functions": functions,
        "activity": _field(rng, 0.4),
        "participation": {**_field(rng, 0.4), "settings_affected": rng.sample(
            ["home", "childcare_school", "community"], rng.randint(0, 2))},
        "environmental_factors": {
            "caregiver_capacity": _env(rng),
            "home_community_environment": _env(rng),
            "service_access_social_determinants": _env(rng),
            "family_physician_or_pediatrician_involved": rng.choice(["yes", "no", "not_documented"]),
            "other_professionals": professionals,
        },
        "personal_factors": {
            "age": f"{rng.randint(2, 6)} years",
            "developmental_factors": developmental,
            "behavioural_profile": {"status": "not_documented", "description": "", "evidence": ""},
            "birth_history": {"status": "not_documented", "description": "", "evidence": ""},
            "other": [],
        },
        "summary": {
            "areas_of_need": [],
            "n_areas_of_need": 0,
            "max_severity_qualifier": 0,
            "n_other_professionals": len(professionals),
            "developmental_factors_present": "none",
            "activity_or_participation_limited": "not_documented",
            "caregiver_capacity_impacted": "not_documented",
            "environmental_or_personal_factors_impacting_care": "not_documented",
            "documentation_gaps": ["birth_history"],
        },
    }
    from .icf_profile import recompute_summary

    profile["summary"] = recompute_summary(profile)
    return profile


def synthetic_cohort(n, start=0):
    from .icf_profile import recompute_summary, render_icf

    cohort = []
    for i in range(n):
        cid = f"SYN{start + i + 1:03d}"
        profile = synthetic_profile(start + i)
        cohort.append((cid, render_icf(cid, profile, recompute_summary(profile))))
    return cohort
