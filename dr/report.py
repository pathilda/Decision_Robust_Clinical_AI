"""Pilot reports for the approval gates (spec Section 12, steps 5 and 8).

    python -m dr.report icf --model medgemma-4b        # parse rate, rendered length, 5 examples
    python -m dr.report pilot --model qwen35-4b        # parse/infeasibility/n_distinct/too long
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter

from .common import JsonlStore, Workspace, add_workspace_argument
from .icf_profile import render_icf


def icf_report(ws, model, examples):
    store = JsonlStore(ws.icf(model))
    records = list(store.records.values())
    if not records:
        print(f"no ICF records for {model} at {ws.icf(model)}")
        return 1
    ok = [r for r in records if r["parse_ok"]]
    lengths = [r["rendered_profile_tokens"] for r in ok]
    flags = Counter(f.split(":")[0] for r in ok for f in r["flags"])
    report = {
        "model": model,
        "n_records": len(records),
        "parse_ok": len(ok),
        "parse_rate": len(ok) / len(records),
        "attempts": dict(Counter(r["attempts"] for r in records)),
        "prompt_too_long": sum(r.get("skipped_reason") == "prompt_too_long" for r in records),
        "rendered_profile_tokens_mean": statistics.fmean(lengths) if lengths else None,
        "rendered_profile_tokens_max": max(lengths) if lengths else None,
        "completion_tokens_max": max((r["completion_tokens"] or 0) for r in ok) if ok else None,
        "flag_counts": dict(flags),
    }
    path = ws.icf(model).parent / "pilot_examples.txt"
    blocks = [
        render_icf(r["client_id"], r["parsed"], r["summary_recomputed"])
        + f"\n(flags: {', '.join(r['flags']) or 'none'})"
        for r in ok[:examples]
    ]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    report["examples_file"] = str(path)
    print(json.dumps(report, indent=2))
    return 0


def _summarize(records):
    records = list(records)
    ok = [r for r in records if r["parse_ok"]]
    return {
        "n": len(records),
        "parse_ok": len(ok),
        "parse_rate": round(len(ok) / len(records), 4) if records else None,
        "prompt_too_long": sum(r.get("skipped_reason") == "prompt_too_long" for r in records),
        "retried": sum((r.get("attempts") or 0) > 1 for r in records),
        "flags": dict(Counter(f.split(":")[0] for r in ok for f in r.get("flags", []))),
    }


def pilot_report(ws, model):
    base = ws.outputs / model
    rows = []
    for path in sorted(base.glob("rating/**/results.jsonl")):
        rows.append({"file": str(path.relative_to(base)), **_summarize(JsonlStore(path).records.values())})
    for path in sorted(base.glob("allocation/**/results.jsonl")):
        store = JsonlStore(path)
        ok = list(store.ok().values())
        rows.append({
            "file": str(path.relative_to(base)),
            **_summarize(store.records.values()),
            "infeasible": sum(not r["feasible"] for r in ok),
            "infeasibility_rate": round(sum(not r["feasible"] for r in ok) / len(ok), 4) if ok else None,
        })
    distinct = Counter()
    statuses = Counter()
    for path in sorted(base.glob("candidates/**/*.json")):
        pool = json.loads(path.read_text(encoding="utf-8"))
        statuses[pool["status"]] += 1
        if pool["status"] == "ok":
            distinct[pool["n_distinct"]] += 1
    for path in sorted(base.glob("tiebreak/**/results.jsonl")):
        records = JsonlStore(path).records.values()
        rows.append({
            "file": str(path.relative_to(base)),
            **_summarize(records),
            "skipped": dict(Counter(r.get("skipped_reason") for r in records if r.get("skipped"))),
        })
    print(json.dumps({
        "model": model,
        "files": rows,
        "candidate_pool_status": dict(statuses),
        "n_distinct_distribution": dict(sorted(distinct.items())),
    }, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    icf = sub.add_parser("icf")
    icf.add_argument("--model", required=True)
    icf.add_argument("--examples", type=int, default=5)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--model", required=True)
    for s in (icf, pilot):
        add_workspace_argument(s, data=False)
    args = p.parse_args(argv)
    ws = Workspace.from_args(args)
    if args.command == "icf":
        return icf_report(ws, args.model, args.examples)
    return pilot_report(ws, args.model)


if __name__ == "__main__":
    sys.exit(main())
