"""Build outputs/excluded_clients.csv (spec Section 3, "Missing ICF profiles").

    python -m dr.exclusions                       # all four models (full study)
    python -m dr.exclusions --models qwen35-4b    # pilot with one model

A client whose ICF profile failed (or is missing) for ANY listed model is excluded from every
model's downstream steps. By default the WHOLE chunk containing such a client is excluded too
(reason "chunk_incomplete"), so every chunk used downstream is complete (all its clients have
a successful profile from every model). --keep-partial-chunks keeps the remaining clients of
such chunks instead. Either way the cohorts are identical across models.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .common import (
    MODEL_KEYS,
    JsonlStore,
    Workspace,
    add_workspace_argument,
    now,
    write_json_atomic,
)
from .context import read_chunks


def build(ws, models, keep_partial_chunks=False):
    chunks = read_chunks(ws)
    stores = {m: JsonlStore(ws.icf(m)) for m in models}
    rows = []
    for chunk_id, clients in chunks.items():
        for client in clients:
            reasons = {}
            for m, store in stores.items():
                record = store.get(client)
                if record is None:
                    reasons[m] = "missing"
                elif not record.get("parse_ok"):
                    reasons[m] = record.get("skipped_reason") or "parse_failed"
            if reasons:
                rows.append(
                    {
                        "client_id": client,
                        "chunk_id": chunk_id,
                        "failed_models": ";".join(sorted(reasons)),
                        "reasons": json.dumps(reasons, sort_keys=True),
                    }
                )
    if not keep_partial_chunks:
        hit = {r["chunk_id"] for r in rows}
        failed = {r["client_id"] for r in rows}
        for chunk_id, clients in chunks.items():
            if chunk_id in hit:
                rows.extend(
                    {
                        "client_id": client,
                        "chunk_id": chunk_id,
                        "failed_models": "",
                        "reasons": json.dumps({"chunk": "chunk_incomplete"}),
                    }
                    for client in clients
                    if client not in failed
                )
    ws.excluded.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["client_id", "chunk_id", "failed_models", "reasons"]).to_csv(
        ws.excluded, index=False
    )
    total = sum(map(len, chunks.values()))
    write_json_atomic(
        ws.excluded.with_suffix(".json"),
        {"models": list(models), "n_clients": total, "n_excluded": len(rows),
         "keep_partial_chunks": bool(keep_partial_chunks), "created_at": now()},
    )
    return rows, total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_KEYS), choices=MODEL_KEYS)
    p.add_argument("--keep-partial-chunks", action="store_true",
                   help="Keep the successful clients of a chunk that has a failed client")
    add_workspace_argument(p)
    args = p.parse_args(argv)
    ws = Workspace.from_args(args)
    rows, total = build(ws, args.models, args.keep_partial_chunks)
    chunks = read_chunks(ws)
    excluded = {r["client_id"] for r in rows}
    complete = [c for c, clients in chunks.items() if not excluded & set(clients)]
    partial = [c for c, clients in chunks.items()
               if excluded & set(clients) and not set(clients) <= excluded]
    print(f"{len(rows)} of {total} clients excluded for models {args.models} -> {ws.excluded}")
    print(f"chunks kept complete: {len(complete)} ({total - len(rows)} clients); "
          f"partially kept: {len(partial)}; fully excluded: "
          f"{len(chunks) - len(complete) - len(partial)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
