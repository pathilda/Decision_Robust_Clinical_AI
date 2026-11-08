"""Step 6: LLM tie-break among distinct OR candidates (spec Section 7.2).

    python -m dr.tiebreak --model <key> --rung <4|5|6> --source <label|distribution|score> \
        --profile <...|all>

One call per chunk per load profile per source. If the pool has one distinct assignment the
LLM is not called (record: skipped=true, skipped_reason="single_candidate"). Otherwise the
distinct candidates are shuffled with a seed derived from (chunk_id, profile, rung, source),
numbered 1..K, and the display -> pool mapping (with the original seeds) is stored.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from .common import (
    RATING_OUTPUTS,
    TIEBREAK_RUNGS,
    JsonlStore,
    Workspace,
    derived_seed,
    model_entry,
    models_config,
    now,
    prompt_version,
)
from .context import cohorts
from .engine import ParseError, Request, run_requests
from .or_bridge import load_profiles
from .prompts import tiebreak_messages
from .rate import add_selection_arguments, parse_profiles
from .schemas import tiebreak_schema, validate


def display_order(chunk_id, profile, rung, source, k):
    order = list(range(k))
    random.Random(derived_seed("tiebreak-shuffle-v1", str(chunk_id), profile, rung, source)).shuffle(order)
    return order


def tiebreak_parser(schema, mapping, pool):
    def parse(obj):
        errors = validate(obj, schema)
        if errors:
            raise ParseError("; ".join(errors[:20]))
        shown = int(obj["chosen_candidate"])
        index = mapping[str(shown)]["candidate_index"]
        return obj, [], {
            "chosen_candidate_index": index,
            "chosen_seeds": pool["candidates"][index]["seeds"],
            "assignment": pool["candidates"][index]["assignment"],
        }

    return parse


def meta(model, rung, source, profile, chunk_id):
    return {
        "model": model,
        "step": "tiebreak",
        "rung": rung,
        "output_type": None,
        "source": source,
        "load_profile": profile,
        "chunk_id": chunk_id,
        "client_id": None,
    }


def static_record(model, revision, rung, source, profile, chunk_id, reason, pool=None):
    record = {
        "key": str(chunk_id),
        **meta(model, rung, source, profile, chunk_id),
        "model_revision": revision,
        "prompt_template_version": prompt_version(),
        "prompt_hash": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "raw_text": None,
        "parsed": None,
        "parse_ok": False,
        "attempts": 0,
        "finish_reason": None,
        "skipped": True,
        "skipped_reason": reason,
        "flags": [],
        "candidates_identity": pool.get("identity") if pool else None,
        "timestamp": now(),
    }
    if reason == "single_candidate":
        record.update(
            parse_ok=True,
            n_distinct=1,
            chosen_candidate_index=0,
            chosen_seeds=pool["candidates"][0]["seeds"],
            assignment=pool["candidates"][0]["assignment"],
        )
    return record


def build(ws, model, revision, rung, source, profile, chunk_cohorts, clinicians, store):
    max_tokens = int(models_config()["max_new_tokens"]["tiebreak"])
    requests, static = [], []
    for chunk_id, cohort in chunk_cohorts.items():
        path = ws.candidates(model, rung, source, profile, chunk_id)
        if not path.is_file():
            static.append(static_record(model, revision, rung, source, profile, chunk_id,
                                        "missing_candidates"))
            continue
        pool = json.loads(path.read_text(encoding="utf-8"))
        if pool["status"] != "ok":
            static.append(static_record(model, revision, rung, source, profile, chunk_id,
                                        f"candidates_{pool['status']}", pool))
            continue
        if pool["client_ids"] != [c for c, _ in cohort]:
            static.append(static_record(model, revision, rung, source, profile, chunk_id,
                                        "candidates_stale_cohort", pool))
            continue
        if pool["n_distinct"] == 1:
            static.append(static_record(model, revision, rung, source, profile, chunk_id,
                                        "single_candidate", pool))
            continue
        k = pool["n_distinct"]
        order = display_order(chunk_id, profile, rung, source, k)
        mapping = {
            str(number): {"candidate_index": index, "seeds": pool["candidates"][index]["seeds"]}
            for number, index in enumerate(order, start=1)
        }
        shown = [pool["candidates"][index]["assignment"] for index in order]
        schema = tiebreak_schema(k)
        requests.append(
            Request(
                key=str(chunk_id),
                messages=tiebreak_messages(rung, source, cohort, clinicians, shown),
                schema=schema,
                max_tokens=max_tokens,
                parse=tiebreak_parser(schema, mapping, pool),
                meta=meta(model, rung, source, profile, chunk_id),
                extra={
                    "n_distinct": k,
                    "display_mapping": mapping,
                    "candidates_identity": pool["identity"],
                },
            )
        )
    # Static records are rewritten only when their status changed (keeps resume idempotent).
    changed = []
    for record in static:
        old = store.get(record["key"])
        if not old or (old.get("skipped_reason"), old.get("candidates_identity")) != (
            record["skipped_reason"], record["candidates_identity"]
        ):
            changed.append(record)
    return requests, changed


def run(ws, model, engine, revision, rungs, sources, profiles, chunk_ids=None, batch_size=512):
    if not profiles:
        raise ValueError("Tie-break requires --profile (or --profile all)")
    chunk_cohorts = cohorts(ws, model, chunk_ids)
    all_profiles = load_profiles()
    summaries = []
    for rung in rungs:
        for source in sources:
            for profile in profiles:
                store = JsonlStore(ws.tiebreak(model, rung, source, profile))
                requests, static = build(ws, model, revision, rung, source, profile,
                                         chunk_cohorts, all_profiles[profile], store)
                store.append(static)
                summary = run_requests(engine, requests, store, revision, batch_size) if requests else {
                    "requested": 0, "reused": 0, "ok": 0, "failed": 0, "prompt_too_long": 0}
                summary.update(step="tiebreak", model=model, rung=rung, source=source,
                               profile=profile, static_records=len(static))
                print(json.dumps(summary))
                summaries.append(summary)
    return summaries


def main(argv=None):
    from .icf import add_model_arguments
    from .run_model import open_engine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_model_arguments(p)
    add_selection_arguments(p, TIEBREAK_RUNGS)
    p.add_argument("--source", "--sources", dest="sources", nargs="+", choices=RATING_OUTPUTS,
                   required=True)
    args = p.parse_args(argv)
    model_entry(args.model)
    profiles = parse_profiles(args.profiles)
    if not profiles:
        p.error("--profile is required")
    ws = Workspace.from_args(args)
    cohorts(ws, args.model, args.chunk_ids)
    engine, revision = open_engine(ws, args.model, args.models_dir)
    try:
        run(ws, args.model, engine, revision, args.rungs, args.sources, profiles,
            args.chunk_ids, args.batch_size)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
