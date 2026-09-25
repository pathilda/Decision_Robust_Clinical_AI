"""Unified command-line interface for inspection and direct offline extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .data import ColumnConfig, DataConfig, inspect_input, load_notes
from .model_client import ModelConfig
from .run_extraction import process_clients
from .settings import MODEL_PRESETS, resolve_model_path


DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/scratch/pathilda/Models"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rolling-profile clinical-target extraction from session notes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["inspect", "extract"],
        help="inspect input or run direct offline extraction",
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
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--batch-size",
        "--concurrency",
        dest="batch_size",
        type=int,
        default=8,
        help=(
            "Maximum sequences vLLM processes concurrently during direct batched generation"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
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


def _model_config(
    args: argparse.Namespace,
    model_name: str,
) -> ModelConfig:
    model_path = resolve_model_path(model_name, args.model_root, args.model_path)
    return ModelConfig(
        model_label=model_name,
        model_id=str(model_path),
        tokenizer_id=str(model_path),
        dtype=MODEL_PRESETS[model_name].dtype,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if args.pipeline == "inspect":
            report = inspect_input(_data_config(args, confirmed=False))
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        model_names = _model_names(args)
        if not args.deidentified_confirmed:
            raise ValueError("Extraction requires --deidentified-confirmed")
        notes = load_notes(_data_config(args, confirmed=True), require_deidentified=True)
        clients = _selected_clients(args, notes)
        summaries: dict[str, dict[str, int]] = {}
        for model_name in model_names:
            summaries[model_name] = process_clients(
                notes=notes,
                selected_clients=clients,
                model_config=_model_config(args, model_name),
                output_root=args.output.expanduser().resolve(),
                batch_size=args.batch_size,
            )
        print("\n>>> Finished")
        print(json.dumps(summaries, indent=2))
        has_failures = any(
            result["failed"] or result["context_overflow"]
            for result in summaries.values()
        )
        return 1 if has_failures else 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
