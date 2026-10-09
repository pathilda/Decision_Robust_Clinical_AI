"""Reusable, independent membership creation with a local Python RNG."""

import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cohort import canonical_source
from .inputs import file_hash, read_table
from .reporting import fresh_directory, write_json


def create_chunks(args):
    if args.chunk_size < 1 or args.seed < 0:
        raise ValueError("Chunk size must be positive and seed nonnegative")
    frame, coverage = canonical_source(
        read_table(args.input, args.sheet, columns=[args.id_column, args.employee_column]),
        args.id_column,
        args.discipline,
        args.employee_column,
    )
    ids = sorted(frame[args.id_column])
    if not ids:
        raise ValueError("No patients remain in the selected cohort")
    random.Random(args.seed).shuffle(ids)
    rows, summary = [], []
    for start in range(0, len(ids), args.chunk_size):
        chunk_id = f"chunk_{start // args.chunk_size + 1:04d}"
        group = ids[start : start + args.chunk_size]
        rows.extend(
            {"patient_id": p, "chunk_id": chunk_id, "within_chunk_order": i}
            for i, p in enumerate(group)
        )
        summary.append(
            {
                "chunk_id": chunk_id,
                "patient_count": len(group),
                "is_remainder": len(group) < args.chunk_size,
            }
        )
    output = fresh_directory(args.output)
    pd.DataFrame(rows).to_csv(output / "chunks.csv", index=False, encoding="utf-8")
    pd.DataFrame(summary).to_csv(output / "chunk_summary.csv", index=False, encoding="utf-8")
    write_json(
        output / "chunk_config.json",
        {
            "schema_version": 1,
            "source_path": str(Path(args.input).resolve()),
            "source_sha256": file_hash(args.input),
            "sheet": args.sheet,
            "id_column": args.id_column,
            "employee_column": args.employee_column,
            "disciplines": args.discipline,
            **coverage,
            "seed": args.seed,
            "rng": "Python random.Random (MT19937), lexicographic ID sort then one shuffle",
            "python_version": platform.python_version(),
            "requested_size": args.chunk_size,
            "eligible_count": len(ids),
            "chunk_count": len(summary),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "membership_sha256": file_hash(output / "chunks.csv"),
            "independent_sample_warning": "Repeated seeds are sensitivity analyses, not independent patient samples",
        },
    )
    return 0
