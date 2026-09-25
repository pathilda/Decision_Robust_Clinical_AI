"""Unified command-line interface for inspection, serving, and extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .data import ColumnConfig, DataConfig, inspect_input, load_notes
from .model_client import ModelConfig
from .run_extraction import process_clients
from .server import ManagedVLLMServer, VLLMServerError
from .settings import MODEL_PRESETS, resolve_model_path


DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/scratch/pathilda/Models"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential clinical-target extraction from session notes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["inspect", "extract", "serve"],
        help="inspect input, run extraction, or start only the model server",
    )
    parser.add_argument("--input", type=Path, help="Excel/CSV/Parquet notes file")
    parser.add_argument("--sheet", default=None, help="Excel sheet name")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--note-column", default="statement2")
    parser.add_argument(
        "--model",
        choices=[*MODEL_PRESETS, "all"],
        default="qwen",
        help="Model preset; 'all' runs the three models sequentially",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the checkpoint path (only valid for one model)",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--client-id", action="append", help="Repeat to select clients")
    parser.add_argument(
        "--max-clients",
        type=int,
        help="Process only the first N clients; omit to process all clients",
    )
    parser.add_argument(
        "--deidentified-confirmed",
        action="store_true",
        help="Required acknowledgement that the input contains no identifying data",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    return parser


def _data_config(args: argparse.Namespace, *, confirmed: bool) -> DataConfig:
    if args.input is None:
        raise ValueError("--input is required for inspect and extract")
    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    return DataConfig(
        input_path=args.input.expanduser().resolve(),
        sheet_name=sheet,
        deidentified_confirmed=confirmed,
        columns=ColumnConfig(client_id=args.id_column, note_text=args.note_column),
    )


def _selected_clients(args: argparse.Namespace, notes) -> list[str]:
    available = list(dict.fromkeys(note.client_id for note in notes))
    selected = list(dict.fromkeys(args.client_id or available))
    unknown = [client_id for client_id in selected if client_id not in available]
    if unknown:
        raise ValueError(f"Unknown --client-id values: {unknown}")
    if args.max_clients is not None:
        if args.max_clients < 1:
            raise ValueError("--max-clients must be at least 1")
        selected = selected[: args.max_clients]
    return selected


def _model_names(args: argparse.Namespace) -> list[str]:
    if args.model_path is not None and args.model == "all":
        raise ValueError("--model-path cannot be combined with --model all")
    return list(MODEL_PRESETS) if args.model == "all" else [args.model]


def _server(args: argparse.Namespace, model_name: str) -> ManagedVLLMServer:
    preset = MODEL_PRESETS[model_name]
    model_path = resolve_model_path(model_name, args.model_root, args.model_path)
    return ManagedVLLMServer(
        preset=preset,
        model_path=model_path,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        startup_timeout=args.startup_timeout,
    )


def _model_config(
    args: argparse.Namespace,
    model_name: str,
    server: ManagedVLLMServer,
) -> ModelConfig:
    return ModelConfig(
        model_label=model_name,
        model_id=str(server.model_path),
        tokenizer_id=str(server.model_path),
        base_url=server.base_url,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        structured_output_mode=MODEL_PRESETS[model_name].structured_output_mode,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.pipeline == "inspect":
            report = inspect_input(_data_config(args, confirmed=False))
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        model_names = _model_names(args)
        if args.pipeline == "serve":
            if len(model_names) != 1:
                raise ValueError("--pipeline serve accepts one model, not --model all")
            server = _server(args, model_names[0])
            server.start()
            try:
                server.wait_until_ready()
                print(f"\nServer ready at {server.base_url}; press Ctrl+C to stop.")
                server.process.wait()
            except KeyboardInterrupt:
                print("\nStopping server...")
            finally:
                server.stop()
            return 0

        if not args.deidentified_confirmed:
            raise ValueError("Extraction requires --deidentified-confirmed")
        notes = load_notes(_data_config(args, confirmed=True), require_deidentified=True)
        clients = _selected_clients(args, notes)
        summaries: dict[str, dict[str, int]] = {}
        for model_name in model_names:
            with _server(args, model_name) as server:
                summaries[model_name] = process_clients(
                    notes=notes,
                    selected_clients=clients,
                    model_config=_model_config(args, model_name, server),
                    output_root=args.output.expanduser().resolve(),
                )
        print("\n>>> Finished")
        print(json.dumps(summaries, indent=2))
        has_failures = any(
            result["failed"] or result["context_overflow"]
            for result in summaries.values()
        )
        return 1 if has_failures else 0
    except (FileNotFoundError, ValueError, VLLMServerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
