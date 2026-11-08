"""Step 2: decision-blind ICF profiles, one call per client (spec Section 4).

    python -m dr.icf --model qwen35-4b [--max-clients 10]

Notes come from the existing loader with the configuration recorded by dr.chunks. The input
contains no class definitions, class scores or caseload information.
"""

from __future__ import annotations

import argparse
import json
import sys

from clinical_target_extraction.src.data import load_clients

from .common import (
    JsonlStore,
    Workspace,
    add_models_dir_argument,
    add_workspace_argument,
    file_sha256,
    models_config,
)
from .context import chunk_of, read_chunks
from .engine import ParseError, Request, run_requests
from .icf_profile import consistency_flags, recompute_summary, render_icf, summary_differences
from .prompts import icf_messages
from .schemas import load, validate


def load_notes(ws):
    from .chunks import loader_config

    if not ws.chunks_meta.is_file():
        raise FileNotFoundError(f"{ws.chunks_meta} not found; run python -m dr.chunks first")
    meta = json.loads(ws.chunks_meta.read_text(encoding="utf-8"))
    if file_sha256(meta["clients"]) != meta["source_sha256"]:
        raise ValueError(f"The client file changed since chunking: {meta['clients']}")
    if not meta.get("deidentified_confirmed"):
        raise ValueError("Chunks were created without --deidentified-confirmed")
    return {c.client_id: c.note_text for c in load_clients(loader_config(meta))}


def icf_parser(client_id, count_text):
    schema = load("icf")

    def parse(obj):
        errors = validate(obj, schema)
        if errors:
            raise ParseError("; ".join(errors[:20]))
        summary = recompute_summary(obj)
        rendered = render_icf(client_id, obj, summary)
        flags = consistency_flags(obj)
        differences = summary_differences(obj["summary"], summary)
        if differences:
            flags.append("summary_recomputed_differs:" + ",".join(differences))
        extra = {
            "summary_recomputed": summary,
            "rendered_profile_tokens": count_text(rendered),
        }
        return obj, flags, extra

    return parse


def build_requests(ws, model, revision, engine, client_filter=None, max_clients=None):
    chunks = read_chunks(ws)
    where = chunk_of(chunks)
    notes = load_notes(ws)
    order = [c for clients in chunks.values() for c in clients]
    missing = [c for c in order if c not in notes]
    if missing:
        raise ValueError(f"Chunked clients missing from the client file: {missing[:10]}")
    if client_filter:
        unknown = set(client_filter) - set(order)
        if unknown:
            raise ValueError(f"Unknown client IDs: {sorted(unknown)}")
        order = [c for c in order if c in set(client_filter)]
    if max_clients:
        order = order[:max_clients]
    max_tokens = int(models_config()["max_new_tokens"]["icf"])
    return [
        Request(
            key=client_id,
            messages=icf_messages(client_id, notes[client_id]),
            schema=load("icf"),
            max_tokens=max_tokens,
            parse=icf_parser(client_id, engine.count_text),
            meta={
                "model": model,
                "step": "icf",
                "rung": None,
                "output_type": None,
                "source": None,
                "load_profile": None,
                "chunk_id": where[client_id],
                "client_id": client_id,
            },
        )
        for client_id in order
    ]


def run(ws, model, engine, revision, client_filter=None, max_clients=None, batch_size=512):
    requests = build_requests(ws, model, revision, engine, client_filter, max_clients)
    summary = run_requests(engine, requests, JsonlStore(ws.icf(model)), revision, batch_size)
    print(json.dumps({"step": "icf", "model": model, **summary}))
    return summary


def add_model_arguments(p):
    p.add_argument("--model", required=True)
    add_models_dir_argument(p)
    p.add_argument("--batch-size", type=int, default=512, help="Requests per persisted batch")
    add_workspace_argument(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_model_arguments(p)
    p.add_argument("--client-id", action="append", default=None)
    p.add_argument("--max-clients", type=int, default=None, help="Pilot: first N chunked clients")
    args = p.parse_args(argv)
    from .run_model import open_engine

    ws = Workspace.from_args(args)
    engine, revision = open_engine(ws, args.model, args.models_dir)
    try:
        run(ws, args.model, engine, revision, args.client_id, args.max_clients, args.batch_size)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
