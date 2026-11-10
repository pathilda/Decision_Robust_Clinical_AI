"""Prompt assembly for every inference (spec Sections 4, 5, 6.1, 7.2).

Order (static first, most variable last, for prefix caching):
SYSTEM: role + ICF reading guide + class definitions [+ class scores] + output instructions
USER:   [r>=2] PURPOSE, [r>=3] GOAL, [r=6] FORMULATION, [r>=5] OPERATIONAL STATE,
        [r>=4] COHORT, then TARGET (rating) / TASK (allocation) / CANDIDATES (tie-break).
"""

from __future__ import annotations

from .common import classes_config, fill, format_score, sha256_text, stable_json, template
from .or_bridge import new_client_limit

BLOCK_SEPARATOR = "\n\n"
RESTRICTIONS = [
    "- Every new client must be assigned to exactly one clinician.",
    "- No clinician can receive more new clients than their limit.",
    # The existing complexity-range OR model enforces no other constraint (see
    # NOTES_existing_code.md); add one plain-language line here for any new constraint.
]


def messages(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_hash(msgs):
    return sha256_text(stable_json(msgs))


# ----------------------------------------------------------------------------- inference 1
def icf_messages(client_id, notes):
    return messages(
        template("icf_system"),
        fill(template("icf_user"), CLIENT_ID=client_id, NOTES=notes),
    )


# ----------------------------------------------------------------------------- system
def class_definitions():
    cfg = classes_config()
    lines = []
    for cid in cfg["ids"]:
        lines.append(f"Class {cid}:")
        lines.extend(f"- {bullet}" for bullet in cfg["definitions"][cid])
        if cfg["therapy_plans"][cid]:
            lines.append(f"- Typical therapy plan: {cfg['therapy_plans'][cid]}")
    if cfg["abbreviations"]:
        listing = ", ".join(f"{k} = {v}" for k, v in cfg["abbreviations"].items())
        lines.append(f"Abbreviations: {listing}.")
    return "\n".join(lines)


def class_scores_block():
    cfg = classes_config()
    listing = ", ".join(f"Class {cid} = {format_score(cfg['scores'][cid])}" for cid in cfg["ids"])
    return fill(template("output_score_classes"), CLASS_SCORE_LIST=listing)


def output_instructions(kind, source=None):
    """kind: label | distribution | score | allocation | tiebreak (with source)."""
    cfg = classes_config()
    if kind in ("label", "distribution", "allocation"):
        return template(f"output_{kind}")
    if kind == "score":
        answer = fill(
            template("output_score_answer"),
            SCORE_MIN=format_score(cfg["min"]),
            SCORE_MAX=format_score(cfg["max"]),
        )
        return class_scores_block() + BLOCK_SEPARATOR + answer
    if kind == "tiebreak":
        if source not in ("label", "distribution", "score"):
            raise ValueError("Tie-break requires a source output type")
        # The source's output-specific context is kept: class scores only when source=score.
        prefix = class_scores_block() + BLOCK_SEPARATOR if source == "score" else ""
        return prefix + template("output_tiebreak")
    raise ValueError(f"Unknown output kind {kind!r}")


def system_prompt(kind, source=None):
    return fill(
        template("rating_system"),
        CLASS_DEFINITIONS=class_definitions(),
        OUTPUT_INSTRUCTIONS=output_instructions(kind, source),
    )


# ----------------------------------------------------------------------------- user blocks
def purpose_block():
    return template("block_purpose")


def goal_block():
    return template("block_goal")


def formulation_block(score_output):
    note = " For this task, s_i is the complexity score you provide." if score_output else ""
    return fill(template("block_formulation"), SCORE_NOTE=note)


def operational_block(clinicians):
    lines = [
        f"- Clinician {c['id']}: current caseload complexity "
        f"{format_score(c['initial_complexity'])}, from {int(c['initial_count'])} current "
        f"clients; can take at most {new_client_limit(c)} new clients."
        for c in clinicians
    ]
    return fill(
        template("block_operational"),
        CLINICIAN_LINES="\n".join(lines),
        RESTRICTION_LINES="\n".join(RESTRICTIONS),
    )


def cohort_block(rendered_profiles, with_target):
    """rendered_profiles: render_icf strings in chunks.xlsx order."""
    return fill(
        template("block_cohort"),
        N=len(rendered_profiles),
        TARGET_SENTENCE=" The target client is one of them." if with_target else "",
        PROFILES=BLOCK_SEPARATOR.join(rendered_profiles),
    )


def target_single_block(rendered_profile):
    return fill(template("block_target_single"), PROFILE=rendered_profile)


def target_cohort_block(client_id):
    return fill(template("block_target_cohort"), CLIENT_ID=client_id)


def task_allocation_block(n):
    return fill(template("block_task_allocation"), N=n)


def candidates_block(candidates, clinicians, chunk_order):
    """candidates: list of {patient: clinician} in display order (Candidate 1..K)."""
    position = {p: i for i, p in enumerate(chunk_order)}
    lines = []
    for number, assignment in enumerate(candidates, start=1):
        lines.append(f"Candidate {number}:")
        for c in clinicians:
            mine = sorted((p for p, j in assignment.items() if j == c["id"]), key=position.get)
            lines.append(f"- Clinician {c['id']}: {', '.join(mine) if mine else 'no new clients'}")
    return fill(template("block_candidates"), CANDIDATE_LINES="\n".join(lines))


def _context_blocks(rung, score_output, clinicians):
    blocks = []
    if rung >= 2:
        blocks.append(purpose_block())
    if rung >= 3:
        blocks.append(goal_block())
    if rung == 6:
        blocks.append(formulation_block(score_output))
    if rung >= 5:
        if clinicians is None:
            raise ValueError("Rungs 5 and 6 require the clinicians of a load profile")
        blocks.append(operational_block(clinicians))
    return blocks


# ----------------------------------------------------------------------------- inference 2/3
def rating_messages(rung, output, target_id, cohort, clinicians=None):
    """cohort: [(client_id, rendered)] for the target's chunk in chunk order."""
    rendered = dict(cohort)
    blocks = _context_blocks(rung, output == "score", clinicians)
    if rung >= 4:
        blocks.append(cohort_block([r for _, r in cohort], with_target=True))
        blocks.append(target_cohort_block(target_id))
    else:
        blocks.append(target_single_block(rendered[target_id]))
    return messages(system_prompt(output), BLOCK_SEPARATOR.join(blocks))


def allocation_messages(rung, cohort, clinicians):
    if rung not in (5, 6):
        raise ValueError("Full allocation runs at rungs 5 and 6 only")
    blocks = _context_blocks(rung, False, clinicians)
    blocks.append(cohort_block([r for _, r in cohort], with_target=False))
    blocks.append(task_allocation_block(len(cohort)))
    return messages(system_prompt("allocation"), BLOCK_SEPARATOR.join(blocks))


def tiebreak_messages(rung, source, cohort, clinicians, candidates):
    """The rating prompt of (rung, source) without TARGET, with tie-break instructions and the
    CANDIDATES block appended. ``clinicians`` orders the candidate listing at every rung but is
    described in an OPERATIONAL STATE block only at rungs 5 and 6."""
    if rung not in (4, 5, 6):
        raise ValueError("Tie-break runs at rungs 4 to 6 only")
    blocks = _context_blocks(rung, source == "score", clinicians if rung >= 5 else None)
    blocks.append(cohort_block([r for _, r in cohort], with_target=False))
    blocks.append(candidates_block(candidates, clinicians, [p for p, _ in cohort]))
    return messages(system_prompt("tiebreak", source), BLOCK_SEPARATOR.join(blocks))
