"""Render one complete example prompt per configuration and the maximum prompt length per
configuration, for human review before any full run (spec Sections 9 and 12, step 7).

    python -m dr.preview --model qwen35-4b                 # real ICF profiles of that model
    python -m dr.preview --synthetic                       # structure only, synthetic profiles

Token counts use the model's own tokenizer and chat template when the checkpoint is available
(--models-dir); otherwise they are labelled as a 4-characters-per-token estimate.
prompts_preview/ can contain clinical text: it is git-ignored and must stay on the cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .common import (
    ALLOCATION_RUNGS,
    PROFILE_IDS,
    RATING_OUTPUTS,
    RUNGS,
    TIEBREAK_RUNGS,
    Workspace,
    add_models_dir_argument,
    add_workspace_argument,
    model_entry,
    models_config,
)
from .or_bridge import load_profiles, new_client_limit, selection_seeds
from .prompts import allocation_messages, icf_messages, rating_messages, tiebreak_messages
from .schemas import allocation_schema, load, tiebreak_schema


class TokenCounter:
    def __init__(self, entry):
        self.method = "estimate_4_chars_per_token"
        self.tokenizer = None
        self.kwargs = dict((entry or {}).get("chat_template_kwargs") or {})
        path = Path(entry["path"]) if entry else None
        if path and (path / "tokenizer_config.json").is_file():
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.method = f"tokenizer:{path.name}"

    def __call__(self, msgs):
        if self.tokenizer is None:
            return sum(len(m["content"]) for m in msgs) // 4
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **self.kwargs
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def as_text(msgs, schema, header):
    return (
        f"# {header}\n\n=== SYSTEM ===\n{msgs[0]['content']}\n\n=== USER ===\n{msgs[1]['content']}\n\n"
        f"=== JSON SCHEMA (guided decoding) ===\n{json.dumps(schema, indent=2)}\n"
    )


def demo_candidates(cohort, clinicians, k):
    """Worst-case-length stand-ins when no real pools exist: k capacity-feasible rotations."""
    ids = [c for c, _ in cohort]
    result = []
    for shift in range(k):
        assignment, counts = {}, {c["id"]: 0 for c in clinicians}
        for i, p in enumerate(ids):
            for step in range(len(clinicians)):
                c = clinicians[(i + shift + step) % len(clinicians)]
                if counts[c["id"]] < new_client_limit(c):
                    assignment[p] = c["id"]
                    counts[c["id"]] += 1
                    break
        result.append(assignment)
    return result


def real_candidates(ws, model, rung, source, profile, chunk_id):
    path = ws.candidates(model, rung, source, profile, chunk_id)
    if not path.is_file():
        return None
    pool = json.loads(path.read_text(encoding="utf-8"))
    if pool.get("status") != "ok" or pool["n_distinct"] < 2:
        return None
    return [c["assignment"] for c in pool["candidates"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--synthetic", action="store_true")
    add_models_dir_argument(p, required=False)
    add_workspace_argument(p, previews=True)
    args = p.parse_args(argv)
    if not args.synthetic and not args.model:
        p.error("--model is required unless --synthetic")
    ws = Workspace.from_args(args)
    entry = None
    if args.model:
        try:
            entry = model_entry(args.model, args.models_dir)
        except ValueError as exc:
            print(f"warning: {exc}; using a length estimate", file=sys.stderr)
    count = TokenCounter(entry)
    profiles = load_profiles()
    cfg = models_config()
    limit = int(cfg["serving"]["max_model_len"])
    new_tokens = cfg["max_new_tokens"]

    if args.synthetic:
        from .fixtures import synthetic_cohort

        chunk_cohorts = {"1": synthetic_cohort(20)}
        notes = {"SYN001": "<SERVICE_PLANNING_TEXT>\nSYNTHETIC NOTE\n</SERVICE_PLANNING_TEXT>\n\n"
                           "<ASSESSMENT_TEXT>\nSYNTHETIC NOTE\n</ASSESSMENT_TEXT>"}
        label = "synthetic"
    else:
        from .context import cohorts
        from .icf import load_notes

        chunk_cohorts = cohorts(ws, args.model)
        notes = load_notes(ws)
        label = args.model
    out = ws.previews / label
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(name, step, rung, kind, profile, lengths, budget, example, schema):
        (out / f"{name}.txt").write_text(as_text(example, schema, name), encoding="utf-8")
        longest = max(lengths)
        rows.append({
            "configuration": name, "step": step, "rung": rung, "output_or_source": kind,
            "profile": profile, "n_prompts": len(lengths), "max_prompt_tokens": longest,
            "max_new_tokens": budget, "max_model_len": limit,
            "fits": longest + budget <= limit, "token_method": count.method,
        })

    first_cohort = next(iter(chunk_cohorts.values()))
    # ICF (decision-blind)
    icf_ids = [c for cohort in chunk_cohorts.values() for c, _ in cohort if c in notes]
    icf_example = icf_messages(icf_ids[0], notes[icf_ids[0]])
    record("icf", "icf", None, None, None, [count(icf_messages(c, notes[c])) for c in icf_ids],
           new_tokens["icf"], icf_example, load("icf"))
    # Ratings
    for rung in RUNGS:
        for output in RATING_OUTPUTS:
            for profile in ([None] if rung <= 4 else PROFILE_IDS):
                clin = profiles[profile] if profile else None
                lengths = [
                    count(rating_messages(rung, output, cid, cohort, clin))
                    for cohort in chunk_cohorts.values() for cid, _ in cohort
                ]
                example = rating_messages(rung, output, first_cohort[0][0], first_cohort, clin)
                name = f"rating_rung{rung}_{output}" + (f"_{profile}" if profile else "")
                record(name, "rating", rung, output, profile, lengths, new_tokens["rating"],
                       example, load(output))
    # Full allocation
    for rung in ALLOCATION_RUNGS:
        for profile in PROFILE_IDS:
            clin = profiles[profile]
            lengths = [count(allocation_messages(rung, cohort, clin)) for cohort in chunk_cohorts.values()]
            schema = allocation_schema([c for c, _ in first_cohort], [c["id"] for c in clin])
            record(f"allocation_rung{rung}_{profile}", "allocation", rung, "allocation", profile,
                   lengths, new_tokens["allocation"], allocation_messages(rung, first_cohort, clin),
                   schema)
    # Tie-break (real pools when present, else K = n_seeds worst-case stand-ins)
    k_max = len(selection_seeds())
    for rung in TIEBREAK_RUNGS:
        for source in RATING_OUTPUTS:
            for profile in PROFILE_IDS:
                clin = profiles[profile]
                lengths, example, example_k = [], None, None
                for chunk_id, cohort in chunk_cohorts.items():
                    cands = None if args.synthetic else real_candidates(
                        ws, args.model, rung, source, profile, chunk_id)
                    cands = cands or demo_candidates(cohort, clin, k_max)
                    msgs = tiebreak_messages(rung, source, cohort, clin, cands)
                    lengths.append(count(msgs))
                    if example is None:
                        example, example_k = msgs, len(cands)
                record(f"tiebreak_rung{rung}_{source}_{profile}", "tiebreak", rung, source, profile,
                       lengths, new_tokens["tiebreak"], example, tiebreak_schema(example_k))
    with (out / "prompt_lengths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    too_long = [r["configuration"] for r in rows if not r["fits"]]
    print(f"wrote {len(rows)} previews to {out} (tokens: {count.method})")
    if too_long:
        print(f"WARNING prompt_too_long risk: {too_long}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
