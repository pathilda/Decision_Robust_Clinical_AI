"""Command-line interface for one-row-per-client treatment classification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .data import (
    CLIENT_ID_COLUMN,
    ClientNote,
    ColumnConfig,
    DataConfig,
    inspect_input,
    load_clients,
)
from .model_client import ModelConfig
from .prompts import PROMPT_STYLES
from .run_classification import process_clients
from .settings import MODEL_PRESETS, resolve_model_path


DEFAULT_MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/scratch/pathilda/Models"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify one combined SP/assessment pre-treatment note per client"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["inspect", "classify"],
        help="validate the workbook or run client-level classification",
    )
    parser.add_argument("--input", type=Path, required=True, help="Excel client workbook")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--id-column", default=CLIENT_ID_COLUMN)
    parser.add_argument("--sp-column", default="SP text")
    parser.add_argument("--assessment-column", default="assessment text")
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
    parser.add_argument(
        "--prompt-style",
        choices=list(PROMPT_STYLES),
        default="standard",
        help="Use the original prompt or request a concise evidence-based rationale",
    )
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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        "--concurrency",
        dest="batch_size",
        type=int,
        default=8,
        help="Maximum independent clients submitted in each direct vLLM batch",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    return parser


def _data_config(args: argparse.Namespace, *, confirmed: bool) -> DataConfig:
    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    return DataConfig(
        input_path=args.input.expanduser().resolve(),
        sheet_name=sheet,
        deidentified_confirmed=confirmed,
        columns=ColumnConfig(
            client_id=args.id_column,
            sp_text=args.sp_column,
            assessment_text=args.assessment_column,
        ),
    )


def _selected_clients(args: argparse.Namespace, clients: list[ClientNote]) -> list[str]:
    available = [client.client_id for client in clients]
    selected = list(dict.fromkeys(args.client_id or available))
    unknown = [client_id for client_id in selected if client_id not in set(available)]
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


def _model_config(args: argparse.Namespace, model_name: str) -> ModelConfig:
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

        if not args.deidentified_confirmed:
            raise ValueError("Classification requires --deidentified-confirmed")
        clients = load_clients(_data_config(args, confirmed=True))
        selected = _selected_clients(args, clients)
        summaries: dict[str, dict[str, int]] = {}
        for model_name in _model_names(args):
            summaries[model_name] = process_clients(
                clients=clients,
                selected_clients=selected,
                model_config=_model_config(args, model_name),
                output_root=args.output.expanduser().resolve(),
                batch_size=args.batch_size,
                prompt_style=args.prompt_style,
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
