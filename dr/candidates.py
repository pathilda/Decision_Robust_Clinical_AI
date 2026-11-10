"""Step 5: OR candidate pools from ratings with the existing seeded solves (spec Section 7.1).

    python -m dr.candidates --model <key> --rung <4|5|6> --source <label|distribution|score> \
        --profile <...|all> --seeds 20

For each chunk and load profile, the existing OR model is solved for 20 selection seeds
(deterministic model for label/score, stochastic SAA model for distribution), duplicates are
removed (canonical form: sorted (client_id, clinician_id) pairs) and the pool is saved with
the seed and objective value of every solution. Rungs 1-3 are accepted too: their pools are
the "OR solves on ratings" that evaluation (Section 10) needs; tie-breaks use rungs 4-6.

Only rating records produced by the current prompt (same prompt hash, i.e. the same cohort,
exclusions, ICF profiles and templates) are used.
"""

from __future__ import annotations

import argparse
import json
import sys

from .common import (
    RATING_OUTPUTS,
    RUNGS,
    JsonlStore,
    Workspace,
    add_gurobi_argument,
    add_workspace_argument,
    apply_gurobi_license,
    classes_config,
    file_sha256,
    now,
    prompt_version,
    sha256_text,
    stable_json,
    write_json_atomic,
)
from .context import cohorts
from .or_bridge import (
    OBJECTIVE,
    Solver,
    canonical_assignment,
    feasible_capacity,
    instance,
    label_weights,
    load_profiles,
    selection_seeds,
)
from .rate import build_requests as rating_requests
from .rate import parse_profiles

ALGORITHM = "dr-candidates-v1"


def chunk_ratings(store, requests):
    """{client: record} if every client in the chunk has a current successful rating."""
    found, problems = {}, []
    for request in requests:
        record = store.get(request.key)
        if record is None or not record.get("parse_ok"):
            problems.append(f"{request.meta['client_id']}:missing_or_failed")
        elif record["prompt_hash"] != request.prompt_hash:
            problems.append(f"{request.meta['client_id']}:stale_prompt")
        else:
            found[request.meta["client_id"]] = record
    return found, problems


def pool(solver, model, rung, source, profile_id, clinicians, chunk_id, ratings, seeds):
    inst = instance(chunk_id, list(ratings), profile_id, clinicians)
    if source == "distribution":
        probabilities = {p: r["probabilities"] for p, r in ratings.items()}
        result = solver.stochastic(inst, probabilities, model, seeds)
        inputs = probabilities
    else:
        if source == "label":
            weights = label_weights({p: r["value"] for p, r in ratings.items()})
        else:
            weights = {p: float(r["value"]) for p, r in ratings.items()}
        result = solver.deterministic(inst, weights, seeds)
        inputs = weights
    distinct = {}
    for solution in result["solutions"]:
        key = stable_json(canonical_assignment(solution["assignment"]))
        entry = distinct.setdefault(
            key,
            {
                "assignment": solution["assignment"],
                "canonical": canonical_assignment(solution["assignment"]),
                "seeds": [],
                "objective_value": solution["objective"],
            },
        )
        entry["seeds"].append(solution["seed"])
    candidates = list(distinct.values())
    index = {stable_json(c["canonical"]): i for i, c in enumerate(candidates)}
    return {
        "status": "ok",
        "policy": result["policy"],
        "objective": OBJECTIVE,
        "optimum": result["optimum"],
        "source_assessments": inputs,
        "n_runs": len(result["solutions"]),
        "n_distinct": len(candidates),
        "candidates": candidates,
        "solutions": [
            {
                "seed": s["seed"],
                "objective_value": s["objective"],
                "candidate_index": index[stable_json(canonical_assignment(s["assignment"]))],
            }
            for s in result["solutions"]
        ],
        **{k: v for k, v in result.items() if k.startswith(("scenario", "n_distinct_"))},
    }


def run(ws, model, rungs, sources, profiles, chunk_ids=None, n_seeds=None):
    chunk_cohorts = cohorts(ws, model, chunk_ids)
    all_profiles = load_profiles()
    seeds = selection_seeds()
    if n_seeds is not None and n_seeds != len(seeds):
        raise ValueError(f"--seeds must equal optimization.n_seeds={len(seeds)} in study.yaml")
    counts = {"written": 0, "reused": 0, "incomplete": 0, "infeasible": 0, "failed": 0}
    with Solver() as solver:
        identity_base = {
            "algorithm": ALGORITHM,
            "seeds": seeds,
            "solver": solver.parameters(),
            "optimization": solver.opt,
            "classes_sha256": classes_config()["sha256"],
            "profiles_sha256": file_sha256(ws_profiles_path()),
        }
        for rung in rungs:
            for source in sources:
                for profile_id in profiles:
                    clinicians = all_profiles[profile_id]
                    rating_profile = profile_id if rung >= 5 else None
                    store = JsonlStore(ws.rating(model, rung, source, rating_profile))
                    requests = rating_requests(
                        model, rung, source, rating_profile, chunk_cohorts,
                        clinicians if rung >= 5 else None,
                    )
                    before = dict(counts)
                    by_chunk = {}
                    for request in requests:
                        by_chunk.setdefault(request.meta["chunk_id"], []).append(request)
                    for chunk_id, chunk_requests in by_chunk.items():
                        path = ws.candidates(model, rung, source, profile_id, chunk_id)
                        ratings, problems = chunk_ratings(store, chunk_requests)
                        header = {
                            "model": model,
                            "rung": rung,
                            "source": source,
                            "load_profile": profile_id,
                            "chunk_id": chunk_id,
                            "client_ids": [r.meta["client_id"] for r in chunk_requests],
                            "rating_prompt_hashes": [r.prompt_hash for r in chunk_requests],
                            "prompt_template_version": prompt_version(),
                            "clinicians": clinicians,
                        }
                        identity = sha256_text(stable_json({**identity_base, **header,
                            "ratings": {p: r["parsed"] for p, r in ratings.items()}}))
                        if path.is_file():
                            saved = json.loads(path.read_text(encoding="utf-8"))
                            if saved.get("identity") == identity and saved["status"] == "ok":
                                counts["reused"] += 1
                                continue
                        if problems:
                            payload = {"status": "incomplete_ratings", "problems": problems}
                            counts["incomplete"] += 1
                        elif not feasible_capacity(list(ratings), clinicians):
                            payload = {"status": "infeasible_capacity"}
                            counts["infeasible"] += 1
                        else:
                            try:
                                payload = pool(solver, model, rung, source, profile_id,
                                               clinicians, chunk_id, ratings, seeds)
                                counts["written"] += 1
                            except (ValueError, RuntimeError) as exc:
                                payload = {"status": "solve_failed", "error": str(exc)}
                                counts["failed"] += 1
                        write_json_atomic(
                            path, {**header, **payload, "identity": identity, "created_at": now()}
                        )
                    print(json.dumps({"step": "candidates", "model": model, "rung": rung,
                                      "source": source, "profile": profile_id,
                                      **{k: counts[k] - before[k] for k in counts}}))
    return counts


def ws_profiles_path():
    from .common import CONFIG_DIR, study_config

    return CONFIG_DIR / study_config()["load_profiles"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rung", "--rungs", dest="rungs", type=int, nargs="+", choices=RUNGS,
                   required=True)
    p.add_argument("--source", "--sources", dest="sources", nargs="+", choices=RATING_OUTPUTS,
                   required=True)
    p.add_argument("--profile", "--profiles", dest="profiles", nargs="+", required=True,
                   choices=["balanced", "unbalanced", "highly_unbalanced", "all"])
    p.add_argument("--seeds", type=int, default=None, help="Must equal study.yaml n_seeds (20)")
    p.add_argument("--chunk-ids", nargs="+", default=None)
    add_workspace_argument(p)
    add_gurobi_argument(p)
    args = p.parse_args(argv)
    apply_gurobi_license(args)
    counts = run(Workspace.from_args(args), args.model, args.rungs, args.sources,
                 parse_profiles(args.profiles), args.chunk_ids, args.seeds)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
