"""CLI for the isolated, local-only OpenDecider inference pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clinical_target_extraction.src.data import (
    CLIENT_ID_COLUMN,
    ClientNote,
    ColumnConfig,
    DataConfig,
    inspect_input,
    load_clients,
)

from .pipeline import process_clients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local OpenDecider treatment-difficulty classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pipeline", required=True, choices=["inspect", "classify"])
    parser.add_argument("--input", type=Path, required=True, help="Excel client workbook")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or zero-based index")
    parser.add_argument("--id-column", default=CLIENT_ID_COLUMN)
    parser.add_argument("--sp-column", default="SP text")
    parser.add_argument("--assessment-column", default="assessment text")
    parser.add_argument("--adapter-path", type=Path, help="Local OpenDecider adapter directory")
    parser.add_argument("--base-path", type=Path, help="Local Qwen base-model directory")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--client-id", action="append", help="Repeat to select clients")
    parser.add_argument("--max-clients", type=int, help="Process only the first N clients")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deidentified-confirmed",
        action="store_true",
        help="Required acknowledgement that the input contains no identifying data",
    )
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
    available_set = set(available)
    selected = list(dict.fromkeys(args.client_id or available))
    unknown = [client_id for client_id in selected if client_id not in available_set]
    if unknown:
        raise ValueError(f"Unknown --client-id values: {unknown}")
    if args.max_clients is not None:
        if args.max_clients < 1:
            raise ValueError("--max-clients must be at least 1")
        selected = selected[: args.max_clients]
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        if args.pipeline == "inspect":
            print(
                json.dumps(
                    inspect_input(_data_config(args, confirmed=False)),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        if not args.deidentified_confirmed:
            raise ValueError("Classification requires --deidentified-confirmed")
        if args.adapter_path is None or args.base_path is None:
            raise ValueError("Classification requires --adapter-path and --base-path")
        clients = load_clients(_data_config(args, confirmed=True))
        summary = process_clients(
            clients=clients,
            selected_clients=_selected_clients(args, clients),
            adapter_path=args.adapter_path,
            base_path=args.base_path,
            output_root=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
        print("\n>>> Finished")
        print(json.dumps({"opendecider": summary}, indent=2))
        return 1 if summary["failed"] else 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
