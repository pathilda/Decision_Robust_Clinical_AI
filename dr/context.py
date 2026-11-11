"""Shared inputs for downstream steps: chunks, exclusions, ICF profiles and cohorts."""

from __future__ import annotations

import json

import pandas as pd

from patient_assignment.cohort import normalize_id

from .common import JsonlStore
from .icf_profile import render_icf

CHUNK_COLUMNS = ["chunk_id", "position", "client_id"]


def read_chunks(ws):
    """{chunk_id: [client_id, ...]} in chunk order, client order = position (1-based)."""
    if not ws.chunks_file.is_file():
        raise FileNotFoundError(f"{ws.chunks_file} not found; run python -m dr.chunks first")
    frame = pd.read_excel(ws.chunks_file, dtype=object, keep_default_na=False)
    if list(frame.columns) != CHUNK_COLUMNS:
        raise ValueError(f"{ws.chunks_file}: expected columns {CHUNK_COLUMNS}")
    frame["client_id"] = frame["client_id"].map(normalize_id)
    frame["chunk_id"] = frame["chunk_id"].map(normalize_id)
    frame["position"] = frame["position"].astype(int)
    # A client may appear in several chunks (labelled chunks are sampled with replacement
    # across chunks; result keys are "<chunk>|<client>"), but never twice in one chunk.
    if frame.duplicated(["chunk_id", "client_id"]).any():
        raise ValueError("chunks.xlsx lists a client more than once in the same chunk")
    chunks = {}
    for chunk_id, group in frame.groupby("chunk_id", sort=False):
        group = group.sort_values("position")
        if list(group["position"]) != list(range(1, len(group) + 1)):
            raise ValueError(f"chunk {chunk_id}: positions must be 1..n")
        chunks[chunk_id] = list(group["client_id"])
    return dict(sorted(chunks.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))


def chunk_of(chunks):
    return {c: chunk for chunk, clients in chunks.items() for c in clients}


def read_exclusions(ws, model):
    """Excluded client IDs; the exclusion list must have been built including ``model``."""
    meta_path = ws.excluded.with_suffix(".json")
    if not ws.excluded.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"{ws.excluded} not found; run python -m dr.exclusions after the ICF step"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if model not in meta["models"]:
        raise ValueError(
            f"Exclusions were built for {meta['models']}, not {model}; rerun python -m "
            "dr.exclusions with every model whose ICF profiles are used"
        )
    frame = pd.read_csv(ws.excluded, dtype=str, keep_default_na=False)
    return set(frame["client_id"]) if len(frame) else set()


def icf_profiles(ws, model):
    """{client_id: (parsed_profile, recomputed_summary)} for successful ICF records."""
    store = JsonlStore(ws.icf(model))
    return {
        r["client_id"]: (r["parsed"], r["summary_recomputed"]) for r in store.ok().values()
    }


def cohorts(ws, model, chunk_ids=None):
    """{chunk_id: [(client_id, render_icf(...)), ...]} after exclusions, in chunk order.

    Model X's ICF profiles are used only by model X.
    """
    chunks = read_chunks(ws)
    if chunk_ids:
        missing = set(chunk_ids) - set(chunks)
        if missing:
            raise ValueError(f"Unknown chunk IDs: {sorted(missing)}")
        chunks = {c: chunks[c] for c in chunks if c in set(chunk_ids)}
    excluded = read_exclusions(ws, model)
    profiles = icf_profiles(ws, model)
    result = {}
    for chunk_id, clients in chunks.items():
        kept = [c for c in clients if c not in excluded]
        if not kept:  # whole chunk excluded, e.g. ICF was run with --max-clients
            continue
        missing = [c for c in kept if c not in profiles]
        if missing:
            raise ValueError(
                f"{model}: no ICF profile for {missing[:10]} in chunk {chunk_id}; "
                "rerun the ICF step and python -m dr.exclusions"
            )
        result[chunk_id] = [(c, render_icf(c, *profiles[c])) for c in kept]
    return result
