"""Spec invariants for prompts, rendering, schemas, parsing and chunking (no LLM, no solver)."""

import json

import pytest

from dr import schemas
from dr.chunks import make_chunks
from dr.common import SCHEMAS_DIR, fill
from dr.engine import ParseError
from dr.fixtures import synthetic_cohort, synthetic_profile
from dr.icf_profile import recompute_summary, render_icf
from dr.or_bridge import check_allocation, load_profiles
from dr.prompts import (
    allocation_messages,
    icf_messages,
    rating_messages,
    system_prompt,
    tiebreak_messages,
)
from dr.rate import rating_parser

COHORT = synthetic_cohort(6)
PROFILES = load_profiles()
CLIN = PROFILES["balanced"]
FORMULA_MARKERS = ("minimize", "x_ij", "sum over", "FORMAL PROBLEM", "L0_j")
CANDS = [
    {c: ("A", "B", "C")[i % 3] for i, (c, _) in enumerate(COHORT)},
    {c: ("B", "C", "A")[i % 3] for i, (c, _) in enumerate(COHORT)},
]


def user(msgs):
    return msgs[1]["content"]


def test_schema_files_match_builders():
    for name, builder in schemas.STATIC.items():
        on_disk = json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert on_disk == builder(), name
    assert list(schemas.load("icf")["properties"])[-1] == "summary"
    for name in ("label", "distribution", "score"):
        assert next(iter(schemas.load(name)["properties"])) == "reasoning"


def test_fill_rejects_missing_and_unused_placeholders():
    assert fill("a {X} {\"json\": 1}", X="b") == 'a b {"json": 1}'
    with pytest.raises(KeyError):
        fill("{X}{Y}", X=1)
    with pytest.raises(KeyError):
        fill("{X}", X=1, Y=2)


def test_icf_prompt_is_decision_blind():
    msgs = icf_messages("C1", "<SERVICE_PLANNING_TEXT>\nnote\n</SERVICE_PLANNING_TEXT>")
    text = msgs[0]["content"] + msgs[1]["content"]
    for banned in ("Class 1", "complexity score", "caseload", "clinician", "1.8"):
        assert banned not in text
    assert "Client ID: C1" in msgs[1]["content"]


def test_rendering_uses_recomputed_summary_and_marks_silence():
    profile = synthetic_profile(3)
    profile["summary"]["n_areas_of_need"] = 99  # model's wrong count must not leak
    summary = recompute_summary(profile)
    text = render_icf("X", profile, summary)
    assert text.startswith("[Client X]\nAge: ")
    assert f"Summary: {summary['n_areas_of_need']} area(s) of need" in text
    assert "99 area" not in text
    nd = [n for n, f in profile["body_functions"].items() if f["status"] == "not_documented"]
    assert nd and "not documented" in text


def test_summary_rules():
    profile = synthetic_profile(1)
    for f in profile["body_functions"].values():
        f.update(status="not_documented", qualifier=9, description="", evidence="")
    profile["body_functions"]["fluency"].update(status="known", qualifier=8)
    profile["body_functions"]["receptive_language"].update(status="queried", qualifier=2)
    profile["body_functions"]["expressive_language"].update(status="no_concern", qualifier=0)
    s = recompute_summary(profile)
    assert s["areas_of_need"] == ["fluency", "receptive_language"]
    assert s["max_severity_qualifier"] == 2
    profile["body_functions"]["receptive_language"].update(status="not_documented", qualifier=9)
    assert recompute_summary(profile)["max_severity_qualifier"] == 8
    cg = profile["environmental_factors"]["caregiver_capacity"]
    cg.update(status="not_documented")
    assert recompute_summary(profile)["caregiver_capacity_impacted"] == "not_documented"
    cg.update(status="known", direction="barrier", magnitude=2)
    assert recompute_summary(profile)["caregiver_capacity_impacted"] == "yes"


@pytest.mark.parametrize("output", ["label", "distribution", "score"])
def test_rung_blocks_are_cumulative(output):
    target = COHORT[2][0]
    headings = ["PURPOSE", "GOAL OF THE CASELOAD MANAGEMENT", "FORMAL PROBLEM", "CLINICIANS",
                "GROUP OF NEW CLIENTS", "TARGET CLIENT"]
    expected = {
        1: ["TARGET CLIENT"],
        2: ["PURPOSE", "TARGET CLIENT"],
        3: ["PURPOSE", "GOAL OF THE CASELOAD MANAGEMENT", "TARGET CLIENT"],
        4: ["PURPOSE", "GOAL OF THE CASELOAD MANAGEMENT", "GROUP OF NEW CLIENTS", "TARGET CLIENT"],
        5: ["PURPOSE", "GOAL OF THE CASELOAD MANAGEMENT", "CLINICIANS", "GROUP OF NEW CLIENTS",
            "TARGET CLIENT"],
        6: headings,
    }
    for rung, wanted in expected.items():
        msgs = rating_messages(rung, output, target, COHORT, CLIN if rung >= 5 else None)
        blocks = [b.split("\n", 1)[0] for b in user(msgs).split("\n\n") if b.split("\n", 1)[0] in headings]
        assert blocks == wanted, rung
        text = user(msgs)
        if rung <= 3:
            assert text.count("[Client ") == 1 and f"[Client {target}]" in text
        else:
            assert text.count("[Client ") == len(COHORT)
            assert text.rstrip().endswith("your answer concerns this client.")
            order = [text.index(f"[Client {c}]") for c, _ in COHORT]
            assert order == sorted(order)
        if rung == 5:
            assert not any(m in text for m in FORMULA_MARKERS)
        has_score_note = "s_i is the complexity score you provide" in text
        assert has_score_note == (rung == 6 and output == "score")


def test_class_scores_only_for_score_output():
    for kind in ("label", "distribution", "allocation"):
        assert "CLASS SCORES" not in system_prompt(kind)
        assert "1.8" not in system_prompt(kind)
    assert "Class 5 = 1.8" in system_prompt("score")
    assert "CLASS SCORES" in system_prompt("tiebreak", "score")
    assert "CLASS SCORES" not in system_prompt("tiebreak", "label")
    for kind in ("label", "distribution", "score", "allocation"):
        assert "Class 1:\n- Client with mild/moderate needs in 1 area" in system_prompt(kind)
        assert "- Typical therapy plan: joint, team or 1:1 (often team/joint)" in system_prompt(kind)
        # Null abbreviations are placeholders and never reach a prompt.
        assert "Abbreviations: HCP = health care provider." in system_prompt(kind)
        assert "None" not in system_prompt(kind) and "KA =" not in system_prompt(kind)
        assert "receptive and expressive language) mean needs in multiple domains" in system_prompt(kind)


def test_no_formulas_at_rung5_anywhere():
    texts = [
        allocation_messages(5, COHORT, CLIN),
        tiebreak_messages(5, "score", COHORT, CLIN, CANDS),
        *(rating_messages(5, o, COHORT[0][0], COHORT, CLIN) for o in ("label", "distribution", "score")),
    ]
    for msgs in texts:
        assert not any(m in user(msgs) for m in FORMULA_MARKERS)
    assert "FORMAL PROBLEM" in user(allocation_messages(6, COHORT, CLIN))


def test_allocation_and_tiebreak_structure():
    alloc = user(allocation_messages(6, COHORT, CLIN))
    assert alloc.rstrip().endswith(f"Assign all {len(COHORT)} clients in the group above to the clinicians.")
    assert "TARGET CLIENT" not in alloc and "The target client is one of them" not in alloc
    tb4 = tiebreak_messages(4, "label", COHORT, CLIN, CANDS)
    text = user(tb4)
    assert "CLINICIANS\n" not in text and "TARGET CLIENT" not in text
    assert text.rstrip().endswith("Choose one candidate.")
    assert "Candidate 1:\n- Clinician A:" in text and "Candidate 2:" in text
    assert "chosen_candidate" in tb4[0]["content"]


def test_candidate_listing_uses_chunk_order():
    from dr.prompts import candidates_block

    order = ["z", "a", "m"]
    text = candidates_block([{"a": "A", "z": "A", "m": "B"}], CLIN, order)
    assert "- Clinician A: z, a" in text and "- Clinician C: no new clients" in text


def test_make_chunks_merges_small_remainder():
    ids = [f"c{i:03d}" for i in range(45)]
    groups = make_chunks(ids, 20, 42, 10)
    assert [len(g) for g in groups] == [20, 25]
    assert sorted(c for g in groups for c in g) == ids
    assert [len(g) for g in make_chunks(ids[:30], 20, 42, 10)] == [20, 10]
    assert make_chunks(ids, 20, 42, 10) == groups  # deterministic


def test_distribution_parser_renormalizes_and_flags():
    parse = rating_parser("distribution")
    base = {"reasoning": "r", "fit_issue": None}
    _, flags, extra = parse({**base, "probabilities": {"1": .1, "2": .2, "3": .3, "4": .2, "5": .19}})
    assert flags == [] and abs(sum(extra["probabilities"].values()) - 1) < 1e-12
    _, flags, extra = parse({**base, "probabilities": {"1": .5, "2": .5, "3": .5, "4": 0, "5": 0}})
    assert flags == ["flag_bad_sum"] and extra["flag_bad_sum"] and extra["probability_sum_raw"] == 1.5
    with pytest.raises(ParseError):
        parse({**base, "probabilities": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}})
    with pytest.raises(ParseError):
        parse({**base, "probabilities": {"1": 1}})


def test_score_parser_clips_and_label_parser_maps():
    _, flags, extra = rating_parser("score")({"reasoning": "", "fit_issue": "x", "complexity_score": 2.3})
    assert flags == ["clipped"] and extra["value"] == 1.8 and extra["complexity_score_raw"] == 2.3
    _, flags, extra = rating_parser("label")({"reasoning": "", "fit_issue": None, "class": 4})
    assert extra == {"value": 4, "class_score": 1.6}
    with pytest.raises(ParseError):
        rating_parser("label")({"reasoning": "", "fit_issue": None, "class": 6})


def test_allocation_feasibility_check():
    ids = [c for c, _ in COHORT]
    ok, violations, assignment = check_allocation(
        [{"client_id": c, "clinician_id": "A"} for c in ids], ids, CLIN)
    assert ok and not violations and set(assignment) == set(ids)
    pairs = [{"client_id": ids[0], "clinician_id": "A"}] * 2 + [
        {"client_id": c, "clinician_id": "B"} for c in ids[2:]]
    ok, violations, _ = check_allocation(pairs, ids, CLIN)
    assert not ok
    assert f"client_assigned_more_than_once:{ids[0]}" in violations
    assert f"client_not_assigned:{ids[1]}" in violations
    tight = [dict(c, capacity=c["initial_count"] + 1) for c in CLIN]
    ok, violations, _ = check_allocation(
        [{"client_id": c, "clinician_id": "A"} for c in ids], ids, tight)
    assert not ok and any(v.startswith("capacity_exceeded:A") for v in violations)


def test_profiles_rescaled_and_valid():
    assert list(PROFILES) == ["balanced", "unbalanced", "highly_unbalanced"]
    assert [c["initial_complexity"] for c in PROFILES["highly_unbalanced"]] == [10.0, 14.0, 18.0]


def test_decoding_schema_drops_number_ranges_only():
    from dr.schemas import decoding_schema, load

    score = decoding_schema(load("score"))
    assert score["properties"]["complexity_score"] == {"type": "number"}
    assert load("score")["properties"]["complexity_score"]["maximum"] == 1.8  # original intact
    probs = decoding_schema(load("distribution"))["properties"]["probabilities"]["properties"]
    assert all(v == {"type": "number"} for v in probs.values())
    icf = load("icf")
    assert decoding_schema(icf) == icf  # integer ranges and enums are kept
