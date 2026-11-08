"""Step 10: connect outputs to the existing evaluation (spec Section 10).

    python -m dr.evaluate --models qwen35-4b qwen35-27b medgemma-4b medgemma-27b \
        --benchmark-file reference.xlsx --benchmark-id-column client_id \
        --benchmark-class-column class --benchmark-name reference
    python -m dr.evaluate --models qwen35-4b --benchmark-rating medgemma-27b:1:label

The benchmark (reference labels, classes 1-5) is explicit, as in the existing proxy analysis; it
is a proxy, not ground truth, unless the file holds clinician labels. Per chunk and load profile
the existing code computes the benchmark optimum and one fixed reference allocation
(benchmark seed from study.yaml); every assignment is evaluated with the existing
proxy_solver.evaluate (proxy regret, exact assignment agreement with the reference).

Outputs in outputs/evaluation/<name>/:
- or_rows.csv          every seeded OR solution on ratings (all rungs/outputs/profiles)
- allocation_rows.csv  LLM full allocations: feasibility reported separately from regret
- tiebreak_rows.csv    chosen candidate + best / mean-random / random / worst pool regret, n_distinct
- accuracy.csv         exact agreement and QWK of label (and distribution argmax) vs benchmark
- summary.csv          means by model x step x rung x output/source x profile
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import pandas as pd

from patient_assignment.cohort import normalize_id
from patient_assignment.inputs import read_table
from patient_assignment.proxy_inputs import agreement

from .common import (
    ALLOCATION_RUNGS,
    PROFILE_IDS,
    RATING_OUTPUTS,
    RUNGS,
    TIEBREAK_RUNGS,
    JsonlStore,
    Workspace,
    add_gurobi_argument,
    add_workspace_argument,
    apply_gurobi_license,
    classes_config,
    derived_seed,
    study_config,
)
from .context import cohorts
from .or_bridge import (
    Solver,
    argmax_class,
    evaluate_against,
    instance,
    label_weights,
    load_profiles,
)


def benchmark_classes(args, ws):
    if args.benchmark_file:
        table = read_table(args.benchmark_file, args.benchmark_sheet)
        result = {}
        for row in table.to_dict("records"):
            cid = normalize_id(row[args.benchmark_id_column])
            if cid in result:
                raise ValueError(f"Duplicate benchmark client {cid}")
            value = int(float(row[args.benchmark_class_column]))
            if value not in classes_config()["ids"]:
                raise ValueError(f"Benchmark class {value!r} for {cid} is not a configured class")
            result[cid] = value
        return lambda profile: result
    model, rung, output = args.benchmark_rating.split(":")
    rung = int(rung)
    if output != "label":
        raise ValueError("--benchmark-rating must use the label output")

    def by_profile(profile):
        store = JsonlStore(ws.rating(model, rung, "label", profile if rung >= 5 else None))
        return {r["client_id"]: int(r["value"]) for r in store.ok().values()}

    return by_profile


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark-file")
    group.add_argument("--benchmark-rating", help="MODEL:RUNG:label")
    p.add_argument("--benchmark-sheet", default=None)
    p.add_argument("--benchmark-id-column", default="client_id")
    p.add_argument("--benchmark-class-column", default="class")
    p.add_argument("--benchmark-name", default=None)
    p.add_argument("--chunk-ids", nargs="+", default=None)
    add_workspace_argument(p)
    add_gurobi_argument(p)
    args = p.parse_args(argv)
    apply_gurobi_license(args)
    ws = Workspace.from_args(args)
    name = args.benchmark_name or (args.benchmark_rating or "file").replace(":", "_")
    out = ws.outputs / "evaluation" / name
    out.mkdir(parents=True, exist_ok=True)
    classes_for = benchmark_classes(args, ws)
    benchmark_seed = int(study_config()["evaluation"]["benchmark_seed"])
    profiles = load_profiles()
    or_rows, alloc_rows, tb_rows, acc_rows, notes = [], [], [], [], []
    references = {}
    with Solver() as solver:

        def reference(chunk_id, profile, clients):
            key = (chunk_id, profile, tuple(clients))
            if key not in references:
                bench = classes_for(profile)
                missing = [c for c in clients if c not in bench]
                if missing:
                    references[key] = None
                    notes.append({"chunk_id": chunk_id, "profile": profile,
                                  "note": f"benchmark missing {len(missing)} clients"})
                else:
                    inst = instance(chunk_id, clients, profile, profiles[profile])
                    bw = label_weights({c: bench[c] for c in clients})
                    optimum, ref = solver.benchmark(inst, bw, benchmark_seed)
                    references[key] = (inst, bw, optimum, ref)
            return references[key]

        def metrics(ctx, assignment):
            inst, bw, optimum, ref = ctx
            return evaluate_against(inst, bw, optimum, assignment, ref)

        for model in args.models:
            chunk_cohorts = cohorts(ws, model, args.chunk_ids)
            # accuracy of point assessments vs benchmark (profile-free benchmark rows only)
            for rung in RUNGS:
                for output in ("label", "distribution"):
                    for profile in ([None] if rung <= 4 else PROFILE_IDS):
                        store = JsonlStore(ws.rating(model, rung, output, profile))
                        bench = classes_for(profile or PROFILE_IDS[0])
                        pairs = []
                        for r in store.ok().values():
                            if r["client_id"] in bench:
                                predicted = (r["value"] if output == "label"
                                             else argmax_class(r["probabilities"]))
                                pairs.append((predicted, bench[r["client_id"]]))
                        if pairs:
                            acc, qwk, status = agreement(
                                [a for a, _ in pairs], [b for _, b in pairs], classes_config()["ids"])
                            acc_rows.append({"model": model, "rung": rung, "output": output,
                                             "profile": profile, "n": len(pairs),
                                             "exact_agreement_pct": acc, "qwk": qwk,
                                             "qwk_status": status})
            for chunk_id, cohort in chunk_cohorts.items():
                clients = [c for c, _ in cohort]
                for profile in PROFILE_IDS:
                    ctx = reference(chunk_id, profile, clients)
                    if ctx is None:
                        continue
                    base = {"model": model, "chunk_id": chunk_id, "profile": profile,
                            "benchmark_optimum": ctx[2]["g"]}
                    # (a) OR on ratings, every seed (existing per-seed evaluation)
                    for rung in RUNGS:
                        for output in RATING_OUTPUTS:
                            path = ws.candidates(model, rung, output, profile, chunk_id)
                            if not path.is_file():
                                continue
                            pool = json.loads(path.read_text(encoding="utf-8"))
                            if pool["status"] != "ok" or pool["client_ids"] != clients:
                                notes.append({**base, "rung": rung, "output": output,
                                              "note": f"pool {pool['status']} or stale cohort"})
                                continue
                            for s in pool["solutions"]:
                                assignment = pool["candidates"][s["candidate_index"]]["assignment"]
                                or_rows.append({**base, "step": "rating_or", "rung": rung,
                                                "output": output, "policy": pool["policy"],
                                                "seed": s["seed"], "n_distinct": pool["n_distinct"],
                                                "source_objective": s["objective_value"],
                                                **metrics(ctx, assignment)})
                    # (b) full allocation
                    for rung in ALLOCATION_RUNGS:
                        record = JsonlStore(ws.allocation(model, rung, profile)).get(str(chunk_id))
                        if not record or not record.get("parse_ok"):
                            alloc_rows.append({**base, "step": "allocation", "rung": rung,
                                               "parse_ok": False, "feasible": None})
                            continue
                        row = {**base, "step": "allocation", "rung": rung, "parse_ok": True,
                               "feasible": record["feasible"],
                               "violations": ";".join(record["violations"])}
                        if record["feasible"]:
                            row.update(metrics(ctx, record["assignment"]))
                        alloc_rows.append(row)
                    # (c) tie-break with pool bounds
                    for rung in TIEBREAK_RUNGS:
                        for source in RATING_OUTPUTS:
                            record = JsonlStore(ws.tiebreak(model, rung, source, profile)).get(str(chunk_id))
                            path = ws.candidates(model, rung, source, profile, chunk_id)
                            if not record or not record.get("parse_ok") or not path.is_file():
                                tb_rows.append({**base, "step": "tiebreak", "rung": rung,
                                                "source": source, "parse_ok": False,
                                                "skipped_reason": (record or {}).get("skipped_reason")})
                                continue
                            pool = json.loads(path.read_text(encoding="utf-8"))
                            regrets = [metrics(ctx, c["assignment"])["proxy_regret"]
                                       for c in pool["candidates"]]
                            pick = random.Random(derived_seed("eval-random", model, chunk_id, profile,
                                                              rung, source)).randrange(len(regrets))
                            chosen = metrics(ctx, record["assignment"])
                            tb_rows.append({**base, "step": "tiebreak", "rung": rung, "source": source,
                                            "parse_ok": True, "skipped": record["skipped"],
                                            "n_distinct": pool["n_distinct"], **chosen,
                                            "best_candidate_regret": min(regrets),
                                            "worst_candidate_regret": max(regrets),
                                            "mean_candidate_regret": statistics.fmean(regrets),
                                            "random_candidate_regret": regrets[pick]})
    tables = {"or_rows.csv": or_rows, "allocation_rows.csv": alloc_rows,
              "tiebreak_rows.csv": tb_rows, "accuracy.csv": acc_rows, "notes.csv": notes}
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    summary = []
    for rows, key in ((or_rows, "output"), (tb_rows, "source"), (alloc_rows, None)):
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        group = ["model", "step", "rung", *([key] if key else []), "profile"]
        for ident, part in frame.groupby(group, dropna=False):
            row = dict(zip(group, ident), n=len(part))
            for column in ("proxy_regret", "exact_assignment_pct", "best_candidate_regret",
                           "mean_candidate_regret", "worst_candidate_regret", "n_distinct"):
                if column in part and part[column].notna().any():
                    row[column + "_mean"] = part[column].mean()
            if "feasible" in part:
                parsed = part[part["parse_ok"] == True]
                row["infeasibility_rate"] = (
                    float((parsed["feasible"] == False).mean()) if len(parsed) else None
                )
            summary.append(row)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"evaluation written to {out} ({len(or_rows)} OR rows, {len(alloc_rows)} allocation rows, "
          f"{len(tb_rows)} tie-break rows, {len(notes)} notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
