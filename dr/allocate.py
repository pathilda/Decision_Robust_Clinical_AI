"""Step 4: full allocation by the LLM, rungs 5 and 6 (spec Section 6).

    python -m dr.allocate --model <key> --rung <5|6> --profile <...|all>

One call per chunk per load profile. Infeasible answers are recorded (feasible=false with the
violated constraints) and never retried; only unparseable output is retried.
"""

from __future__ import annotations

import argparse
import json
import sys

from .common import ALLOCATION_RUNGS, JsonlStore, Workspace, model_entry, models_config
from .context import cohorts
from .engine import ParseError, Request, run_requests
from .or_bridge import check_allocation, load_profiles
from .prompts import allocation_messages
from .rate import add_selection_arguments, parse_profiles
from .schemas import allocation_schema, validate


def allocation_parser(client_ids, clinicians, schema):
    def parse(obj):
        errors = validate(obj, schema)
        if errors:
            raise ParseError("; ".join(errors[:20]))
        feasible, violations, assignment = check_allocation(obj["assignment"], client_ids, clinicians)
        flags = [] if feasible else ["infeasible"]
        return obj, flags, {"feasible": feasible, "violations": violations, "assignment": assignment}

    return parse


def build_requests(model, rung, profile, chunk_cohorts, clinicians):
    max_tokens = int(models_config()["max_new_tokens"]["allocation"])
    requests = []
    for chunk_id, cohort in chunk_cohorts.items():
        client_ids = [c for c, _ in cohort]
        schema = allocation_schema(client_ids, [c["id"] for c in clinicians])
        requests.append(
            Request(
                key=str(chunk_id),
                messages=allocation_messages(rung, cohort, clinicians),
                schema=schema,
                max_tokens=max_tokens,
                parse=allocation_parser(client_ids, clinicians, schema),
                meta={
                    "model": model,
                    "step": "allocation",
                    "rung": rung,
                    "output_type": "allocation",
                    "source": None,
                    "load_profile": profile,
                    "chunk_id": chunk_id,
                    "client_id": None,
                },
            )
        )
    return requests


def run(ws, model, engine, revision, rungs, profiles, chunk_ids=None, batch_size=512):
    if not profiles:
        raise ValueError("Full allocation requires --profile (or --profile all)")
    chunk_cohorts = cohorts(ws, model, chunk_ids)
    all_profiles = load_profiles()
    summaries = []
    for rung in rungs:
        for profile in profiles:
            requests = build_requests(model, rung, profile, chunk_cohorts, all_profiles[profile])
            store = JsonlStore(ws.allocation(model, rung, profile))
            summary = run_requests(engine, requests, store, revision, batch_size)
            keys = {r.key for r in requests}
            done = [r for k, r in store.ok().items() if k in keys]
            summary.update(
                step="allocation", model=model, rung=rung, profile=profile,
                infeasible=sum(not r["feasible"] for r in done),
            )
            print(json.dumps(summary))
            summaries.append(summary)
    return summaries


def main(argv=None):
    from .icf import add_model_arguments
    from .run_model import open_engine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_model_arguments(p)
    add_selection_arguments(p, ALLOCATION_RUNGS)
    args = p.parse_args(argv)
    model_entry(args.model)
    profiles = parse_profiles(args.profiles)
    if not profiles:
        p.error("--profile is required")
    ws = Workspace.from_args(args)
    cohorts(ws, args.model, args.chunk_ids)
    engine, revision = open_engine(ws, args.model, args.models_dir)
    try:
        run(ws, args.model, engine, revision, args.rungs, profiles, args.chunk_ids, args.batch_size)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
