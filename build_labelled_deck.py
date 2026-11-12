"""Clinician-labelled study deck (reveal.js + Plotly, one self-contained HTML file).

Two steps, so that only aggregates leave the cluster:

1. ``collect`` (on the cluster, repository root): reads the clinician labels, the labelled
   chunks, the rating / candidate outputs and dr.evaluate's tables, and writes a small JSON
   of aggregates only (counts, means, CIs; no client IDs, no text from notes).

       python build_labelled_deck.py collect \
           --labels-file /home/pathilda/Anonymization/Labeled_Sample.xlsx \
           --data-dir /home/pathilda/Anonymization/Labeled_chunks \
           --outputs-dir /scratch/pathilda/dr_outputs_labeled_od \
           --models medgemma-4b qwen35-4b qwen35-27b --score-as-uniform-benchmark \
           --out /scratch/pathilda/dr_outputs_labeled_od/deck/deck_data.json

2. ``render`` (anywhere; standard library only): puts the JSON into the deck template.

       python build_labelled_deck.py render --data deck_data.json --out slides.html            [--exclude-outputs score] [--inline-libs]

   The page loads reveal.js (cdnjs) and Plotly (jsdelivr) from CDNs. ``--inline-libs`` downloads them once and
   embeds them, so the file also works offline (about 5 MB).

Run ``python -m dr.evaluate ... --benchmark-name clinician`` first; ``collect`` reads
outputs/evaluation/clinician/{or_rows,allocation_rows}.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "presentations" / "labelled_study" / "template.html"
PLACEHOLDER = "/*__DECK_DATA__*/null"
LIBS = {
    "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.1.0/reset.min.css": "style",
    "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.1.0/reveal.min.css": "style",
    "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.1.0/reveal.min.js": "script",
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js": "script",
}
MODEL_NAMES = {
    "medgemma-4b": ("MedGemma 4B", "MG-4B"),
    "medgemma-27b": ("MedGemma 27B", "MG-27B"),
    "qwen35-4b": ("Qwen3.5 4B", "Q-4B"),
    "qwen35-27b": ("Qwen3.5 27B", "Q-27B"),
}
# Two-sided 95% t quantiles by degrees of freedom; 1.96 beyond the table.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
       17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def mean_ci(values):
    """Mean and 95% t half-width over independent units (None when undefined)."""
    v = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if not v:
        return None, None, 0
    m = statistics.fmean(v)
    if len(v) < 2:
        return m, None, 1
    t = T95.get(len(v) - 1) or next((T95[k] for k in sorted(T95) if k >= len(v) - 1), 1.96)
    return m, t * statistics.stdev(v) / len(v) ** 0.5, len(v)


def confidence_name(value):
    import re

    text = re.sub(r"\s*confidence\s*", " ", str(value), flags=re.IGNORECASE).strip()
    return text.title() if text else "Missing"


# ----------------------------------------------------------------------------- collect
def collect(args):
    sys.path.insert(0, str(ROOT))
    import pandas as pd

    from dr.common import (
        ALLOCATION_RUNGS,
        CONFIG_DIR,
        PROFILE_IDS,
        RATING_OUTPUTS,
        RUNGS,
        JsonlStore,
        Workspace,
        classes_config,
        models_config,
        read_yaml,
        study_config,
    )
    from dr.labels import as_class, blank, find_column
    from dr.or_bridge import argmax_class
    from patient_assignment.cohort import normalize_id
    from patient_assignment.inputs import read_table
    from patient_assignment.proxy_inputs import agreement

    ws = Workspace(args.data_dir, args.outputs_dir)
    classes = classes_config()
    ids, scores = classes["ids"], classes["scores"]

    # ---- labels
    frame = read_table(args.labels_file, args.sheet)
    id_col = find_column(frame, args.id_column, r"client.*id", "id")
    class_col = find_column(frame, args.class_column, r"class", "class")
    conf_col = find_column(frame, args.confidence_column, r"confiden", "confidence")
    n_rows = len(frame)
    frame = frame[~frame[id_col].map(blank)].copy()
    frame["client_id"] = frame[id_col].map(normalize_id)
    frame = frame.drop_duplicates("client_id", keep="first")
    frame["class"] = frame[class_col].map(lambda v: None if blank(v) else as_class(v))
    frame["confidence"] = frame[conf_col].map(lambda v: "Missing" if blank(v) else confidence_name(v))
    label_of = {c: int(k) for c, k in zip(frame["client_id"], frame["class"]) if k in ids}
    rank = {"Very Low": 0, "Low": 1, "Mid": 2, "Medium": 2, "Moderate": 2, "High": 3,
            "Very High": 4, "Missing": 9}
    conf_order = sorted(frame["confidence"].unique(), key=lambda c: (rank.get(c, 5), c))

    chunks = pd.read_excel(ws.chunks_file, dtype=object)
    chunks["client_id"] = chunks["client_id"].map(normalize_id)
    chunks["chunk_id"] = chunks["chunk_id"].map(normalize_id)
    used_ids = set(chunks["client_id"])

    def stats(part):
        cls = part["class"].map(lambda k: str(int(k)) if k in ids else "missing")
        return {
            "n": len(part),
            "class": {str(k): int((cls == str(k)).sum()) for k in ids},
            "class_missing": int((cls == "missing").sum()),
            "confidence": {c: int((part["confidence"] == c).sum()) for c in conf_order},
            "cross": {str(k): {c: int(((cls == str(k)) & (part["confidence"] == c)).sum())
                               for c in conf_order} for k in ids},
        }

    meta = json.loads(ws.chunks_meta.read_text(encoding="utf-8")) if ws.chunks_meta.is_file() else {}
    chunk_ids = list(dict.fromkeys(chunks["chunk_id"]))
    labels = {
        "n_rows": n_rows,
        "n_icf_all": meta.get("n_eligible", len(used_ids)),
        "conf_order": conf_order,
        "all": stats(frame),
        "used": stats(frame[frame["client_id"].isin(used_ids)]),
        "chunks": {
            cid: {str(k): int((part["client_id"].map(label_of) == k).sum()) for k in ids}
            for cid, part in chunks.groupby("chunk_id", sort=False)
        },
        "chunk_size": int(chunks.groupby("chunk_id").size().max()),
        "usage": {str(t): n for t, n in sorted(Counter(Counter(chunks["client_id"]).values()).items())},
    }

    # ---- evaluation tables
    eval_dir = ws.outputs / "evaluation" / args.benchmark_name
    def table(name):
        path = eval_dir / name
        return pd.read_csv(path) if path.is_file() and path.stat().st_size > 1 else pd.DataFrame()

    orr, alloc = table("or_rows.csv"), table("allocation_rows.csv")
    if orr.empty:
        print(f"warning: no or_rows.csv in {eval_dir}; run dr.evaluate first", file=sys.stderr)
    n_expected = len(chunk_ids) * len(PROFILE_IDS)

    def predicted(record, output):
        if output == "label":
            return int(record["value"])
        if output == "distribution":
            return argmax_class(record["probabilities"])
        value = float(record["value"])
        return min(ids, key=lambda k: (abs(scores[k] - value), k))

    def unit_means(part):
        if part.empty:
            return part
        return part.groupby(["chunk_id", "profile"], as_index=False)[
            [c for c in ("proxy_regret", "exact_assignment_pct", "n_distinct") if c in part]].mean()

    def or_stats(units):
        """Mean and 95% CI over (chunk, profile) units."""
        out = {"n_units": len(units)}
        if units.empty:
            return out
        out["regret_mean"], out["regret_ci"], _ = mean_ci(units["proxy_regret"])
        out["assign_mean"], out["assign_ci"], _ = mean_ci(units["exact_assignment_pct"])
        if "n_distinct" in units:
            out["distinct_mean"] = float(units["n_distinct"].mean())
        return out

    def accuracy(pairs):
        if not pairs:
            return {"n_pairs": 0}
        acc, qwk, _ = agreement([a for a, _ in pairs], [b for _, b in pairs], ids)
        return {"n_pairs": len(pairs), "exact_pct": float(acc),
                "qwk": None if qwk is None or math.isnan(float(qwk)) else float(qwk),
                "mae": statistics.fmean(abs(a - b) for a, b in pairs)}

    def profile_part(units, profile):
        return units[units.profile == profile] if not units.empty else units

    def alloc_stats(rows):
        parsed = rows[rows.parse_ok == True]
        feasible = parsed[parsed.feasible == True]
        return {"n_alloc": len(rows), "n_parsed": len(parsed), "n_feasible": len(feasible),
                **or_stats(feasible)}

    benchmark_output = "score" if args.score_as_uniform_benchmark else None
    benchmark_values = Counter()
    cells = []
    for model in args.models:
        if model in args.placeholder_models:
            continue  # shown as "results pending" in the deck
        for rung in RUNGS:
            for output in RATING_OUTPUTS:
                pairs, failed, total = {}, 0, 0
                for profile in [None] if rung <= 4 else PROFILE_IDS:
                    path = ws.rating(model, rung, output, profile)
                    if not path.is_file():
                        continue
                    for record in JsonlStore(path).records.values():
                        total += 1
                        if not record.get("parse_ok"):
                            failed += 1
                            continue
                        if output == benchmark_output:
                            benchmark_values[round(float(record["value"]), 4)] += 1
                        if record.get("client_id") in label_of:
                            pairs.setdefault(profile, []).append(
                                (predicted(record, output), label_of[record["client_id"]]))
                all_pairs = [x for v in pairs.values() for x in v]
                part = orr[(orr.model == model) & (orr.rung == rung) & (orr.output == output)] \
                    if not orr.empty else orr
                units = unit_means(part)
                cell = {"model": model, "rung": rung, "output": output,
                        "n_ratings": total, "n_failed": failed,
                        **accuracy(all_pairs), **or_stats(units), "n_expected": n_expected}
                cell["by_profile"] = {
                    p: {**or_stats(profile_part(units, p)),
                        **accuracy(pairs.get(None if rung <= 4 else p, [])),
                        "n_expected": len(chunk_ids)}
                    for p in PROFILE_IDS}
                statuses = Counter()
                for chunk_id in chunk_ids:
                    for profile in PROFILE_IDS:
                        path = ws.candidates(model, rung, output, profile, chunk_id)
                        statuses[json.loads(path.read_text(encoding="utf-8"))["status"]
                                 if path.is_file() else "missing"] += 1
                cell["statuses"] = dict(statuses)
                cells.append(cell)
        for rung in ALLOCATION_RUNGS:
            part = alloc[(alloc.model == model) & (alloc.rung == rung)] if not alloc.empty else alloc
            cell = {"model": model, "rung": rung, "output": "allocation", "n_expected": n_expected,
                    "n_alloc": len(part)}
            if not part.empty:
                cell.update(alloc_stats(part))
                cell["by_profile"] = {p: {**alloc_stats(part[part.profile == p]),
                                          "n_expected": len(chunk_ids)} for p in PROFILE_IDS}
            cells.append(cell)

    uniform = None
    if benchmark_output:
        part = orr[orr.output == benchmark_output] if not orr.empty else orr
        cells = [c for c in cells if c["output"] != benchmark_output]
        if part.empty:
            print("warning: no score OR rows; no uniform benchmark", file=sys.stderr)
        else:
            per = part.groupby(["model", "rung", "chunk_id", "profile"])["proxy_regret"].mean()
            spread = float(per.groupby(["chunk_id", "profile"]).agg(lambda v: v.max() - v.min()).max())
            units = unit_means(part)
            uniform = {**or_stats(units), "max_spread": spread,
                       "rating_values": {str(k): n for k, n in benchmark_values.items()},
                       "by_profile": {p: or_stats(profile_part(units, p)) for p in PROFILE_IDS}}
            if len(benchmark_values) != 1 or spread > 1e-6:
                print(f"warning: score ratings are not all equal ({dict(benchmark_values.most_common(5))}; regret "
                      f"spread {spread:.3g}); the 'equal-complexity benchmark' is not valid",
                      file=sys.stderr)

    optimum = {}
    if not orr.empty:
        bench = orr.drop_duplicates(["chunk_id", "profile"])
        optimum = {p: float(g["benchmark_optimum"].mean()) for p, g in bench.groupby("profile")}

    profiles_raw = read_yaml(CONFIG_DIR / study_config()["load_profiles"])["profiles"]
    opt = study_config()["optimization"]
    data = {
        "synthetic": False,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "benchmark": args.benchmark_name,
        "models": [{"key": m, "name": MODEL_NAMES.get(m, (m, m))[0],
                    "short": MODEL_NAMES.get(m, (m, m))[1],
                    "hf_id": models_config()["models"][m]["hf_id"],
                    "placeholder": m in args.placeholder_models} for m in args.models],
        "pending_models": [MODEL_NAMES.get(m, (m, m))[0] for m in args.pending_models],
        "classes": [{"id": k, "score": scores[k], "definition": " ".join(classes["definitions"][k]),
                     "plan": classes["therapy_plans"][k]} for k in ids],
        "profiles": {p: profiles_raw[p]["clinicians"] for p in PROFILE_IDS},
        "optimization": {"n_seeds": opt["n_seeds"], "scenario_count": opt["scenario_count"],
                         "time_limit": opt["time_limit"],
                         "tolerance": opt["optimality_tolerance"]},
        "labels": labels,
        "benchmark_optimum": optimum,
        "uniform_benchmark": uniform,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(cells)} cells, labels n={labels['all']['n']}, "
          f"used n={labels['used']['n']})")
    return 0


# ----------------------------------------------------------------------------- render
def render(args):
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.exclude_outputs:
        data["excluded_outputs"] = sorted(set(args.exclude_outputs))
        data["cells"] = [c for c in data["cells"] if c["output"] not in data["excluded_outputs"]]
    page = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise SystemExit(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, payload)
    if args.inline_libs:
        for url, kind in LIBS.items():
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read().decode("utf-8")
            if kind == "style":
                tag = f'<link rel="stylesheet" href="{url}">'
                page = page.replace(tag, f"<style>{body}</style>")
            else:
                tag = f'<script src="{url}"></script>'
                page = page.replace(tag, "<script>" + body.replace("</script", "<\\/script") + "</script>")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}" + (" (synthetic preview data)" if data.get("synthetic") else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="aggregate results into deck_data.json (cluster)")
    c.add_argument("--labels-file", type=Path, required=True)
    c.add_argument("--sheet", default=None)
    c.add_argument("--id-column", default="Client AlayaCare Client ID")
    c.add_argument("--class-column", default=None)
    c.add_argument("--confidence-column", default=None)
    c.add_argument("--data-dir", type=Path, required=True, help="labelled chunks data dir")
    c.add_argument("--outputs-dir", type=Path, required=True)
    c.add_argument("--benchmark-name", default="clinician")
    c.add_argument("--models", nargs="+", required=True)
    c.add_argument("--placeholder-models", nargs="*", default=[],
                   help="models listed in --models whose results are not ready: shown as pending")
    c.add_argument("--pending-models", nargs="*", default=["medgemma-27b"],
                   help="models not in the deck yet (named in caveats and next steps)")
    c.add_argument("--score-as-uniform-benchmark", action="store_true",
                   help="score ratings are all identical (xgrammar range bug): drop the score "
                        "output and use its OR results as the equal-complexity benchmark")
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render", help="build the HTML deck from deck_data.json")
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--exclude-outputs", nargs="*", default=[], choices=["label", "distribution", "score"],
                   help="leave these rating outputs out of every slide (e.g. score)")
    r.add_argument("--inline-libs", action="store_true", help="embed reveal.js and Plotly")
    args = p.parse_args(argv)
    return collect(args) if args.command == "collect" else render(args)


if __name__ == "__main__":
    sys.exit(main())
