"""Resumable checkpoints and flat result tables for client classification."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from .data import CLIENT_ID_COLUMN
from .schemas import PROBABILITY_COLUMNS, PROBABILITY_STYLES, schema_for_prompt_style

RESULT_COLUMNS = [
    "model",
    CLIENT_ID_COLUMN,
    "prompt_style",
    "brief_reasoning",
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

    def materialize(
        self,
        selected_clients: list[str],
        *,
        prompt_style: str = "standard",
        expected_fingerprints: dict[str, str] | None = None,
    ) -> None:
        """Write one result row per selected client with a valid checkpoint."""

        latest = self.latest_validated()
        with_probabilities = prompt_style in PROBABILITY_STYLES
        columns = list(RESULT_COLUMNS)
        if with_probabilities:
            position = columns.index("treatment_category")
            columns[position:position] = [
                "confidence",
                *PROBABILITY_COLUMNS.values(),
                "probability_source",
                "probabilities_calibrated",
                "probability_contract_version",
            ]
        rows = []
        for client_id in selected_clients:
            record = latest.get(client_id)
            if record is None:
                continue
            if expected_fingerprints is not None and (
                client_id not in expected_fingerprints
                or record.get("request_fingerprint") != expected_fingerprints[client_id]
            ):
                continue
            probability_fields = {}
            if with_probabilities:
                # Revalidate saved distributions; derive table fields from the canonical output.
                output = schema_for_prompt_style(prompt_style).model_validate(
                    record["model_output"]
                )
                probabilities = output.probabilities.model_dump(by_alias=True)
                probability_fields = {
                    "confidence": output.confidence,
                    **{
                        column: probabilities[category]
                        for category, column in PROBABILITY_COLUMNS.items()
                    },
                    "probability_source": record["probability_source"],
                    "probabilities_calibrated": record["probabilities_calibrated"],
                    "probability_contract_version": record["probability_contract_version"],
                    "treatment_category": output.treatment_category,
                    "brief_reasoning": getattr(output, "brief_reasoning", None),
                }
            rows.append(
                {
                    "model": record["model"],
                    CLIENT_ID_COLUMN: client_id,
                    "prompt_style": record.get("prompt_style", "standard"),
                    "brief_reasoning": record.get("brief_reasoning"),
                    "treatment_category": record["treatment_category"],
                    "source_row": record["source_row"],
                    "input_sha256": record["input_sha256"],
                    "input_tokens": record["input_tokens"],
                    "completed_at": record["completed_at"],
                    **probability_fields,
                }
            )
        frame = pd.DataFrame(rows, columns=columns)
        with self._lock:
            self._write_csv_atomic(self.root / "client_classifications.csv", frame)
            self._write_excel_atomic(self.root / "client_classifications.xlsx", frame)
            self._write_parquet_if_available(self.root / "client_classifications.parquet", frame)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    # Preserve confidence-first/category-last inside probability model_output.
                    sort_keys=payload.get("probability_source") != "llm_self_reported",
                )
                + "\n"
            )
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
