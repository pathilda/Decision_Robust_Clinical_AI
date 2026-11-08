"""Step 1: fixed chunks shared by every model, rung and output type (spec Section 3).

    python -m dr.chunks --clients <existing client file> --chunk-size 20 --seed 42 \
        --data-dir <DATA_DIR> --deidentified-confirmed

Writes <DATA_DIR>/chunks.xlsx and <DATA_DIR>/chunks_meta.json.

Clients are loaded with the existing loader (clinical_target_extraction.src.data.load_clients:
first row per normalized client ID, optional PT/OT/SLP discipline filter). IDs are sorted, then
shuffled once with random.Random(seed) (as in the existing chunking), and cut into consecutive
chunks; a last chunk with fewer than 10 clients is merged into the previous chunk.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import pandas as pd

from clinical_target_extraction.src.data import (
    CLIENT_ID_COLUMN,
    ColumnConfig,
    DataConfig,
    load_clients,
)
from patient_assignment.cohort import DISCIPLINES, EMPLOYEE_COLUMN

from .common import (
    Workspace,
    add_workspace_argument,
    file_sha256,
    now,
    study_config,
    write_json_atomic,
)


def make_chunks(client_ids, size, seed, merge_below):
    if size < 1:
        raise ValueError("chunk size must be positive")
    ids = sorted(client_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate client IDs")
    random.Random(seed).shuffle(ids)
    groups = [ids[i : i + size] for i in range(0, len(ids), size)]
    if len(groups) > 1 and len(groups[-1]) < merge_below:
        groups[-2].extend(groups.pop())
    return groups


def loader_config(meta):
    """The existing loader configuration recorded at chunking time (reused by dr.icf)."""
    return DataConfig(
        input_path=Path(meta["clients"]),
        sheet_name=meta["sheet"],
        deidentified_confirmed=meta["deidentified_confirmed"],
        columns=ColumnConfig(**meta["columns"]),
        disciplines=meta["disciplines"],
        employee_column=meta["employee_column"],
    )


def main(argv=None):
    cfg = study_config()["chunks"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clients", type=Path, required=True, help="Existing client Excel workbook")
    p.add_argument("--sheet", default=None, help="Sheet name or zero-based index")
    p.add_argument("--id-column", default=CLIENT_ID_COLUMN)
    p.add_argument("--sp-column", default="SP text")
    p.add_argument("--assessment-column", default="assessment text")
    p.add_argument("--discipline", action="append", choices=DISCIPLINES, default=[])
    p.add_argument("--employee-column", default=EMPLOYEE_COLUMN)
    p.add_argument("--chunk-size", type=int, default=int(cfg["size"]))
    p.add_argument("--seed", type=int, default=int(cfg["seed"]))
    p.add_argument(
        "--deidentified-confirmed",
        action="store_true",
        help="Required: confirms the workbook is de-identified (checked again by dr.icf)",
    )
    add_workspace_argument(p, outputs=False)
    args = p.parse_args(argv)
    if not args.deidentified_confirmed:
        print("error: pass --deidentified-confirmed (the notes are sent to local LLMs)", file=sys.stderr)
        return 2
    ws = Workspace.from_args(args)
    out = ws.chunks_file
    sheet = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet
    meta = {
        "clients": str(args.clients.resolve()),
        "sheet": sheet,
        "columns": {
            "client_id": args.id_column,
            "sp_text": args.sp_column,
            "assessment_text": args.assessment_column,
        },
        "disciplines": args.discipline,
        "employee_column": args.employee_column,
        "deidentified_confirmed": True,
    }
    coverage = {}
    clients = load_clients(loader_config(meta), require_deidentified=True, coverage_out=coverage)
    groups = make_chunks(
        [c.client_id for c in clients], args.chunk_size, args.seed, int(cfg["merge_remainder_below"])
    )
    rows = [
        {"chunk_id": index, "position": position, "client_id": client}
        for index, group in enumerate(groups, start=1)
        for position, client in enumerate(group, start=1)
    ]
    frame = pd.DataFrame(rows, columns=["chunk_id", "position", "client_id"])
    if out.is_file():
        existing = pd.read_excel(out, dtype=object, keep_default_na=False)
        same = existing.astype(str).equals(frame.astype(str))
        if same:
            print(f"{out} already exists with identical chunks; nothing to do")
            return 0
        print(f"error: {out} exists with DIFFERENT chunks; chunks are fixed for the study. "
              "Move it away explicitly if a new study is intended.", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="chunks", index=False)
    meta.update(
        source_sha256=file_sha256(args.clients),
        cohort_coverage=coverage,
        chunk_size=args.chunk_size,
        seed=args.seed,
        merge_remainder_below=int(cfg["merge_remainder_below"]),
        rng="Python random.Random (MT19937); IDs sorted, then shuffled once",
        python_version=platform.python_version(),
        n_clients=len(rows),
        n_chunks=len(groups),
        chunk_sizes=[len(g) for g in groups],
        chunks_sha256=file_sha256(out),
        created_at=now(),
    )
    write_json_atomic(ws.chunks_meta, meta)
    print(json.dumps({k: meta[k] for k in ("n_clients", "n_chunks", "chunk_sizes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
