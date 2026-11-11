"""Class-stratified chunks of clinician-labelled clients, sampled with replacement across chunks.

    python -m dr.label_chunks --labels-file /path/Labeled_Sample.xlsx \
        --icf-outputs-dir /scratch/me/dr_outputs --data-dir /path/to/Labeled_chunks \
        --n-chunks 5 --chunk-size 20 --seed 42

Eligible clients: labelled with a valid class AND a successful ICF profile from every model
(read from --icf-outputs-dir). Sampling:
1. Over all n_chunks x chunk_size slots, each class gets slots proportional to its share of the
   eligible clients (largest remainder).
2. Each class's slots are spread over the chunks as evenly as possible, so every chunk follows
   the class distribution as closely as its size allows (rare classes rotate across chunks).
3. Within a chunk a client appears at most once; across chunks clients repeat. Within each
   class the least-used clients are taken first (ties broken by the seed), so all eligible
   clients are used about equally.
4. Positions inside each chunk are shuffled (seeded), so class order carries no signal.

Writes <data-dir>/chunks.xlsx (chunk_id, position, client_id; chunk IDs L1..Ln) and
<data-dir>/chunks_meta.json, the same format the pipeline reads for the main chunks.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from patient_assignment.cohort import normalize_id
from patient_assignment.inputs import read_table

from .common import MODEL_KEYS, Workspace, classes_config, file_sha256, now, write_json_atomic
from .labels import DEFAULT_ID_COLUMN, as_class, blank, find_column, icf_status


def largest_remainder(shares, total):
    """Integers proportional to ``shares`` summing to ``total`` (ties: larger share first)."""
    whole = sum(shares.values())
    exact = {k: total * v / whole for k, v in shares.items()}
    result = {k: int(x) for k, x in exact.items()}
    order = sorted(exact, key=lambda k: (-(exact[k] - result[k]), -shares[k], k))
    for k in order[: total - sum(result.values())]:
        result[k] += 1
    return result


def plan_counts(class_sizes, n_chunks, chunk_size, rng):
    """{chunk_index: {class: count}}: proportional overall, spread evenly across chunks."""
    totals = largest_remainder(class_sizes, n_chunks * chunk_size)
    base = {k: totals[k] // n_chunks for k in totals}
    extra = {k: totals[k] % n_chunks for k in totals}
    plan = {j: dict(base) for j in range(n_chunks)}
    room = {j: chunk_size - sum(base.values()) for j in range(n_chunks)}
    # Largest remainders first, each to the chunks with the most room left (seeded ties).
    for k in sorted(extra, key=lambda k: (-extra[k], k)):
        chunks = sorted(range(n_chunks), key=lambda j: (-room[j], rng.random()))[: extra[k]]
        for j in chunks:
            plan[j][k] += 1
            room[j] -= 1
    if any(room.values()):
        raise RuntimeError(f"Could not fill every chunk exactly: {room}")
    for j, counts in plan.items():
        for k, count in counts.items():
            if count > class_sizes[k]:
                raise ValueError(
                    f"Chunk {j + 1} needs {count} clients of class {k} but only "
                    f"{class_sizes[k]} exist; use a smaller --chunk-size"
                )
    return plan


def sample(by_class, n_chunks, chunk_size, seed):
    rng = random.Random(seed)
    sizes = {k: len(v) for k, v in by_class.items()}
    plan = plan_counts(sizes, n_chunks, chunk_size, rng)
    used = Counter()
    chunks = []
    for j in range(n_chunks):
        members = []
        for k in sorted(plan[j]):
            pool = sorted(by_class[k], key=lambda c: (used[c], rng.random()))
            picked = pool[: plan[j][k]]
            used.update(picked)
            members.extend(picked)
        rng.shuffle(members)
        chunks.append(members)
    return chunks, plan, used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels-file", type=Path, required=True)
    p.add_argument("--sheet", default=None)
    p.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    p.add_argument("--class-column", default=None)
    p.add_argument("--icf-outputs-dir", type=Path, required=True,
                   help="Outputs dir holding <model>/icf/icf.jsonl (the main study's)")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="New data dir for the labelled chunks (chunks.xlsx is written here)")
    p.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    p.add_argument("--n-chunks", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="L", help="Chunk ID prefix (default L -> L1, L2, ...)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing chunks.xlsx")
    args = p.parse_args(argv)
    ws = Workspace(args.data_dir, args.icf_outputs_dir)
    if ws.chunks_file.exists() and not args.force:
        p.error(f"{ws.chunks_file} exists; pass --force to overwrite (results keyed to the old "
                "chunks would no longer match)")

    frame = read_table(args.labels_file, args.sheet)
    id_col = find_column(frame, args.id_column, r"client.*id", "id")
    class_col = find_column(frame, args.class_column, r"class", "class")
    frame = frame[~frame[id_col].map(blank)].copy()
    frame["client_id"] = frame[id_col].map(normalize_id)
    frame = frame.drop_duplicates("client_id", keep="first")
    frame["class"] = frame[class_col].map(lambda v: None if blank(v) else as_class(v))
    valid = set(classes_config()["ids"])
    labelled = frame[frame["class"].isin(valid)]
    ids = list(labelled["client_id"])
    ok = pd.Series(True, index=labelled.index)
    for model in args.models:
        status = icf_status(ws, model, ids)
        if status is None:
            p.error(f"No ICF file for {model} at {ws.icf(model)}")
        ok &= labelled["client_id"].map(status).eq("ok")
    eligible = labelled[ok]
    print(f"Labelled clients: {len(frame)}; valid class: {len(labelled)}; "
          f"ICF ok for all {len(args.models)} models (eligible): {len(eligible)}")

    by_class = {int(k): sorted(g["client_id"]) for k, g in eligible.groupby("class")}
    chunks, plan, used = sample(by_class, args.n_chunks, args.chunk_size, args.seed)

    rows = []
    for j, members in enumerate(chunks, start=1):
        rows.extend({"chunk_id": f"{args.prefix}{j}", "position": i, "client_id": c}
                    for i, c in enumerate(members, start=1))
    ws.data.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(ws.chunks_file, index=False)

    class_of = dict(zip(eligible["client_id"], eligible["class"].astype(int)))
    table = pd.DataFrame(
        {f"{args.prefix}{j + 1}": {f"Class {k}": plan[j][k] for k in sorted(plan[j])}
         for j in range(args.n_chunks)}
    )
    table["all chunks"] = table.sum(axis=1)
    table["all %"] = (100 * table["all chunks"] / table["all chunks"].sum()).round(1)
    sizes = Counter(class_of.values())
    table["eligible n"] = [sizes[int(r.split()[1])] for r in table.index]
    table["eligible %"] = (100 * table["eligible n"] / len(eligible)).round(1)
    print("\nClass counts per chunk vs. the eligible clients:")
    print(table.to_string())
    usage = Counter(used.values())
    print("\nTimes each eligible client is used: "
          + ", ".join(f"{t}x: {usage[t]} clients" for t in sorted(usage))
          + (f", 0x: {len(eligible) - len(used)} clients" if len(used) < len(eligible) else ""))

    write_json_atomic(ws.chunks_meta, {
        "created_at": now(),
        "kind": "labelled_stratified_with_replacement",
        "labels_file": str(args.labels_file.resolve()),
        "labels_sha256": file_sha256(args.labels_file),
        "id_column": id_col,
        "class_column": class_col,
        "icf_outputs_dir": str(args.icf_outputs_dir.resolve()),
        "models": args.models,
        "n_eligible": len(eligible),
        "eligible_class_counts": {str(k): v for k, v in sorted(sizes.items())},
        "n_chunks": args.n_chunks,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "class_counts_per_chunk": {f"{args.prefix}{j + 1}": {str(k): v for k, v in plan[j].items()}
                                   for j in range(args.n_chunks)},
        "chunks": {f"{args.prefix}{j}": members for j, members in enumerate(chunks, start=1)},
    })
    print(f"\nWrote {ws.chunks_file} and {ws.chunks_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
