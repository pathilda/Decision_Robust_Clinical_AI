"""Incremental audit logs, checkpoints, and analysis-table materialization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


class OutputStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registries = root / "registries"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registries.mkdir(parents=True, exist_ok=True)

    def write_run_config(self, payload: dict[str, Any]) -> None:
        self._write_json_atomic(self.root / "run_config.json", payload)

    def append_raw(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.root / "raw_responses.jsonl", payload)

    def append_validated(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.root / "validated_sessions.jsonl", {"record_type": "validated", **payload}
        )

    def invalidate_from(self, client_id: str, session_index: int, reason: str) -> None:
        """Append a durable tombstone so stale downstream checkpoints cannot reappear."""

        self._append_jsonl(
            self.root / "validated_sessions.jsonl",
            {
                "record_type": "invalidation",
                "client_id": client_id,
                "from_session_index": session_index,
                "reason": reason,
            },
        )

    def append_failure(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.root / "failures.jsonl", payload)

    def save_registry(self, client_id: str, payload: dict[str, Any]) -> None:
        self._write_json_atomic(self.registries / f"{safe_filename(client_id)}.json", payload)

    def latest_validated(self) -> dict[tuple[str, int], dict[str, Any]]:
        records: dict[tuple[str, int], dict[str, Any]] = {}
        path = self.root / "validated_sessions.jsonl"
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("record_type") == "invalidation":
                        client_id = str(record["client_id"])
                        from_session = int(record["from_session_index"])
                        for key in list(records):
                            if key[0] == client_id and key[1] >= from_session:
                                del records[key]
                        continue
                    key = (str(record["client_id"]), int(record["session_index"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid checkpoint line {line_number} in {path}: {exc}"
                    ) from exc
                records[key] = record
        return records

    def materialize_parquet(self) -> None:
        records = self.latest_validated().values()
        session_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        for record in records:
            model_output = record["model_output"]
            model = record["model"]
            for target in record["prior_structured_record"]["target_records"]:
                session_rows.append(
                    _session_row(
                        model,
                        record,
                        target["target_id"],
                        target["substantively_treated"],
                        target["performance_observed"],
                        target["treatment_evidence"],
                        target["performance_evidence"],
                        source=target["source"],
                    )
                )

            if int(record["session_index"]) == 1:
                continue

            for assessment in model_output["target_assessments"]:
                session_rows.append(
                    _session_row(
                        model,
                        record,
                        assessment["target_id"],
                        assessment["substantively_treated"],
                        assessment["performance_observed"],
                        assessment["treatment_evidence"],
                        assessment["performance_evidence"],
                        source="registered_assessment",
                    )
                )
                for comparison in assessment["comparisons"]:
                    comparison_rows.append(
                        {
                            "model": model,
                            "client_id": record["client_id"],
                            "target_id": assessment["target_id"],
                            "current_session_index": record["session_index"],
                            "prior_session_index": comparison["prior_session_index"],
                            "comparable": comparison["comparable"],
                            "better": comparison["better"],
                            "worse": comparison["worse"],
                            "current_evidence": _json_text(comparison["current_evidence"]),
                            "prior_evidence": _json_text(comparison["prior_evidence"]),
                            "comparison_basis": comparison["comparison_basis"],
                        }
                    )

        self._write_parquet_atomic(
            self.root / "session_targets.parquet",
            pd.DataFrame(session_rows, columns=SESSION_COLUMNS),
        )
        self._write_parquet_atomic(
            self.root / "pairwise_comparisons.parquet",
            pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS),
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
    def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)


SESSION_COLUMNS = [
    "model",
    "client_id",
    "session_index",
    "note_id",
    "target_id",
    "substantively_treated",
    "performance_observed",
    "treatment_evidence",
    "performance_evidence",
    "source",
]
COMPARISON_COLUMNS = [
    "model",
    "client_id",
    "target_id",
    "current_session_index",
    "prior_session_index",
    "comparable",
    "better",
    "worse",
    "current_evidence",
    "prior_evidence",
    "comparison_basis",
]


def _session_row(
    model: str,
    record: dict[str, Any],
    target_id: str,
    treated: int,
    observed: int,
    treatment_evidence: list[str],
    performance_evidence: list[str],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "client_id": record["client_id"],
        "session_index": record["session_index"],
        "note_id": record["note_id"],
        "target_id": target_id,
        "substantively_treated": treated,
        "performance_observed": observed,
        "treatment_evidence": _json_text(treatment_evidence),
        "performance_evidence": _json_text(performance_evidence),
        "source": source,
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("client_id cannot be converted to a safe registry filename")
    return cleaned
