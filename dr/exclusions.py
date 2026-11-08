"""Build outputs/excluded_clients.csv (spec Section 3, "Missing ICF profiles").

    python -m dr.exclusions                       # all four models (full study)
    python -m dr.exclusions --models qwen35-4b    # pilot with one model

A client whose ICF profile failed (or is missing) for ANY listed model is excluded from every
model's downstream steps; chunks are otherwise unchanged, so cohorts stay identical.
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


def build(ws, models):
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
    ws.excluded.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["client_id", "chunk_id", "failed_models", "reasons"]).to_csv(
        ws.excluded, index=False
    )
    total = sum(map(len, chunks.values()))
    write_json_atomic(
        ws.excluded.with_suffix(".json"),
        {"models": list(models), "n_clients": total, "n_excluded": len(rows), "created_at": now()},
    )
    return rows, total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_KEYS), choices=MODEL_KEYS)
    add_workspace_argument(p)
    args = p.parse_args(argv)
    ws = Workspace.from_args(args)
    rows, total = build(ws, args.models)
    print(f"{len(rows)} of {total} clients excluded for models {args.models} -> {ws.excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
