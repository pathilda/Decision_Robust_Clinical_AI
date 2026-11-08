"""Step 3: rating inference, one call per target client (spec Section 5).

    python -m dr.rate --model <key> --rung <1..6> --output <label|distribution|score> \
        [--profile <balanced|unbalanced|highly_unbalanced|all>]

--rung and --output accept lists. Rungs 1-4 ignore --profile (one call per client, reused under
every load profile); rungs 5-6 require it and run once per client per profile.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from .common import (
    PROFILE_IDS,
    RATING_OUTPUTS,
    RUNGS,
    JsonlStore,
    Workspace,
    classes_config,
    model_entry,
    models_config,
)
from .context import cohorts
from .engine import ParseError, Request, run_requests
from .or_bridge import load_profiles
from .prompts import rating_messages
from .schemas import load, validate

SUM_TOLERANCE = (0.98, 1.02)


def _relaxed_score_schema():
    schema = json.loads(json.dumps(load("score")))
    schema["properties"]["complexity_score"] = {"type": "number"}
    return schema


def rating_parser(output):
    cfg = classes_config()
    schema = load(output)
    check_schema = _relaxed_score_schema() if output == "score" else schema

    def parse(obj):
        errors = validate(obj, check_schema)
        if errors:
            raise ParseError("; ".join(errors[:20]))
        flags, extra = [], {}
        if output == "label":
            cls = int(obj["class"])
            extra = {"value": cls, "class_score": cfg["scores"][cls]}
        elif output == "distribution":
            raw = {str(i): float(obj["probabilities"][str(i)]) for i in cfg["ids"]}
            total = math.fsum(raw.values())
            if not total > 0:
                raise ParseError("probabilities sum to zero; cannot renormalize")
            normalized = {k: v / total for k, v in raw.items()}
            bad = not SUM_TOLERANCE[0] <= total <= SUM_TOLERANCE[1]
            if bad:
                flags.append("flag_bad_sum")
            extra = {
                "probabilities_raw": raw,
                "probability_sum_raw": total,
                "probabilities": normalized,
                "flag_bad_sum": bad,
            }
        else:
            raw = float(obj["complexity_score"])
            if not math.isfinite(raw):
                raise ParseError("complexity_score is not finite")
            value = min(max(raw, cfg["min"]), cfg["max"])
            clipped = value != raw
            if clipped:
                flags.append("clipped")
            extra = {"complexity_score_raw": raw, "value": value, "clipped": clipped}
        return obj, flags, extra

    return parse


def configurations(rungs, outputs, profiles):
    """[(rung, output, profile-or-None)] in run order."""
    result = []
    for rung in rungs:
        for output in outputs:
            if rung <= 4:
                result.append((rung, output, None))
            else:
                if not profiles:
                    raise ValueError(f"Rung {rung} requires --profile (or --profile all)")
                result.extend((rung, output, p) for p in profiles)
    return result


def build_requests(model, rung, output, profile, chunk_cohorts, clinicians=None):
    max_tokens = int(models_config()["max_new_tokens"]["rating"])
    parse = rating_parser(output)
    requests = []
    for chunk_id, cohort in chunk_cohorts.items():
        for client_id, _ in cohort:
            requests.append(
                Request(
                    key=f"{chunk_id}|{client_id}",
                    messages=rating_messages(rung, output, client_id, cohort, clinicians),
                    schema=load(output),
                    max_tokens=max_tokens,
                    parse=parse,
                    meta={
                        "model": model,
                        "step": "rating",
                        "rung": rung,
                        "output_type": output,
                        "source": None,
                        "load_profile": profile,
                        "chunk_id": chunk_id,
                        "client_id": client_id,
                    },
                )
            )
    return requests


def run(ws, model, engine, revision, rungs, outputs, profiles, chunk_ids=None, batch_size=512):
    chunk_cohorts = cohorts(ws, model, chunk_ids)
    all_profiles = load_profiles()
    summaries = []
    for rung, output, profile in configurations(rungs, outputs, profiles):
        clinicians = all_profiles[profile] if profile else None
        requests = build_requests(model, rung, output, profile, chunk_cohorts, clinicians)
        store = JsonlStore(ws.rating(model, rung, output, profile))
        summary = run_requests(engine, requests, store, revision, batch_size)
        summary.update(step="rating", model=model, rung=rung, output=output, profile=profile)
        print(json.dumps(summary))
        summaries.append(summary)
    return summaries


def parse_profiles(values):
    if not values:
        return []
    if "all" in values:
        return list(PROFILE_IDS)
    return list(dict.fromkeys(values))


def add_selection_arguments(p, rung_choices, rung_default=None):
    p.add_argument("--rung", "--rungs", dest="rungs", type=int, nargs="+", choices=rung_choices,
                   required=rung_default is None, default=rung_default)
    p.add_argument("--profile", "--profiles", dest="profiles", nargs="+",
                   choices=[*PROFILE_IDS, "all"], default=None)
    p.add_argument("--chunk-ids", nargs="+", default=None, help="Pilot: restrict to these chunks")


def main(argv=None):
    from .icf import add_model_arguments
    from .run_model import open_engine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_model_arguments(p)
    add_selection_arguments(p, RUNGS)
    p.add_argument("--output", "--outputs", dest="outputs", nargs="+", choices=RATING_OUTPUTS,
                   required=True)
    args = p.parse_args(argv)
    model_entry(args.model)  # validate the key before loading anything
    profiles = parse_profiles(args.profiles)
    configurations(args.rungs, args.outputs, profiles)  # validate --profile before loading
    ws = Workspace.from_args(args)
    cohorts(ws, args.model, args.chunk_ids)  # validate inputs before loading the model
    engine, revision = open_engine(ws, args.model, args.models_dir)
    try:
        run(ws, args.model, engine, revision, args.rungs, args.outputs, profiles,
            args.chunk_ids, args.batch_size)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
