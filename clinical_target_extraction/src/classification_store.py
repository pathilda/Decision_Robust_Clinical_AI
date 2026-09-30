"""Resumable checkpoints and flat result tables for client classification."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .data import CLIENT_ID_COLUMN


RESULT_COLUMNS = [
    "model",
    CLIENT_ID_COLUMN,
    "treatment_category",
    "source_row",
    "input_sha256",
    "input_tokens",
    "completed_at",
]


class ClassificationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def write_run_config(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._write_json_atomic(self.root / "run_config.json", payload)

    def append_raw(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self.root / "raw_responses.jsonl", payload)

    def append_validated(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(
                self.root / "validated_classifications.jsonl",
                {"record_type": "validated", **payload},
            )

    def append_failure(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append_jsonl(self.root / "failures.jsonl", payload)

    def latest_validated(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            records: dict[str, dict[str, Any]] = {}
            path = self.root / "validated_classifications.jsonl"
            if not path.exists():
                return records
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        client_id = str(record["client_id"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid checkpoint line {line_number} in {path}: {exc}"
                        ) from exc
                    records[client_id] = record
            return records

    def materialize(self, selected_clients: list[str]) -> None:
        """Write one result row per selected client with a valid checkpoint."""

        latest = self.latest_validated()
        rows = []
        for client_id in selected_clients:
            record = latest.get(client_id)
            if record is None:
                continue
            rows.append(
                {
                    "model": record["model"],
                    CLIENT_ID_COLUMN: client_id,
                    "treatment_category": record["treatment_category"],
                    "source_row": record["source_row"],
                    "input_sha256": record["input_sha256"],
                    "input_tokens": record["input_tokens"],
                    "completed_at": record["completed_at"],
                }
            )
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        with self._lock:
            self._write_csv_atomic(self.root / "client_classifications.csv", frame)
            self._write_excel_atomic(self.root / "client_classifications.xlsx", frame)
            self._write_parquet_if_available(
                self.root / "client_classifications.parquet", frame
            )

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)

    @staticmethod
    def _write_excel_atomic(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        frame.to_excel(temporary, index=False)
        os.replace(temporary, path)

    @staticmethod
    def _write_parquet_if_available(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        try:
            frame.to_parquet(temporary, index=False)
        except ImportError:
            return
        os.replace(temporary, path)
