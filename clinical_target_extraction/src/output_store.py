"""Incremental checkpoints and rolling-profile analysis tables."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd


class OutputStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registries = root / "registries"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registries.mkdir(parents=True, exist_ok=True)
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
                self.root / "validated_sessions.jsonl",
                {"record_type": "validated", **payload},
            )

    def invalidate_from(self, client_id: str, session_index: int, reason: str) -> None:
        with self._lock:
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
        with self._lock:
            self._append_jsonl(self.root / "failures.jsonl", payload)

    def save_registry(self, client_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._write_json_atomic(
                self.registries / f"{safe_filename(client_id)}.json",
                payload,
            )

    def latest_validated(self) -> dict[tuple[str, int], dict[str, Any]]:
        with self._lock:
            return self._latest_validated_unlocked()

    def _latest_validated_unlocked(self) -> dict[tuple[str, int], dict[str, Any]]:
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

    def materialize_tables(self) -> None:
        with self._lock:
            records = sorted(
                self._latest_validated_unlocked().values(),
                key=lambda record: (
                    str(record["client_id"]),
                    int(record["session_index"]),
                ),
            )
            self._materialize(records)

    def _materialize(self, records: list[dict[str, Any]]) -> None:
        session_rows: list[dict[str, Any]] = []
        change_rows: list[dict[str, Any]] = []
        profiles: dict[tuple[str, int], dict[str, Any]] = {}

        for record in records:
            profile = record.get("session_profile")
            if not isinstance(profile, dict):
                # Old architecture checkpoints are deliberately not interpreted.
                continue
            key = (str(record["client_id"]), int(record["session_index"]))
            profiles[key] = profile

            for target in profile["targets"]:
                session_rows.append(_session_row(record, target))

        for record in records:
            profile = record.get("session_profile")
            if not isinstance(profile, dict):
                continue
            current_session = int(record["session_index"])
            if current_session == 1:
                continue
            client_id = str(record["client_id"])
            previous = profiles.get((client_id, current_session - 1), {"targets": []})
            previous_by_id = {
                target["target_id"]: target for target in previous.get("targets", [])
            }
            for target in profile["targets"]:
                prior = previous_by_id.get(target["target_id"])
                change_rows.append(_change_row(record, target, prior))

        session_frame = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
        change_frame = pd.DataFrame(change_rows, columns=CHANGE_COLUMNS)
        self._write_analysis_table(self.root / "session_targets.parquet", session_frame)
        self._write_analysis_table(self.root / "target_changes.parquet", change_frame)
        # Keep the historical filename usable, but its scope is now explicitly adjacent-only.
        self._write_analysis_table(
            self.root / "pairwise_comparisons.parquet",
            change_frame,
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
    def _write_analysis_table(path: Path, frame: pd.DataFrame) -> None:
        csv_path = path.with_suffix(".csv")
        csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
        frame.to_csv(csv_temporary, index=False)
        os.replace(csv_temporary, csv_path)

        try:
            parquet_temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_parquet(parquet_temporary, index=False)
            os.replace(parquet_temporary, path)
        except ImportError:
            return


SESSION_COLUMNS = [
    "model",
    "client_id",
    "session_index",
    "note_id",
    "target_id",
    "canonical_label",
    "definition",
    "aliases",
    "verbatim_evidence",
    "evidence_source_session",
    "substantively_treated",
    "performance_observed",
    "change_from_previous",
    "carried_forward",
    "newly_added",
]

CHANGE_COLUMNS = [
    "model",
    "client_id",
    "target_id",
    "current_session_index",
    "prior_session_index",
    "comparison_scope",
    "change_from_previous",
    "comparable",
    "better",
    "worse",
    "improved",
    "stable",
    "worsened",
    "not_assessed",
    "carried_forward",
    "newly_added",
    "current_performance_observed",
    "prior_performance_observed",
    "current_evidence_source_session",
    "current_verbatim_evidence",
    "prior_verbatim_evidence",
]


def _session_row(record: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": record["model"],
        "client_id": record["client_id"],
        "session_index": record["session_index"],
        "note_id": record["note_id"],
        "target_id": target["target_id"],
        "canonical_label": target["canonical_label"],
        "definition": target["definition"],
        "aliases": _json_text(target["aliases"]),
        "verbatim_evidence": _json_text(target["verbatim_evidence"]),
        "evidence_source_session": target["evidence_source_session"],
        "substantively_treated": target["substantively_treated"],
        "performance_observed": target["performance_observed"],
        "change_from_previous": target["change_from_previous"],
        "carried_forward": target["carried_forward"],
        "newly_added": target["newly_added"],
    }


def _change_row(
    record: dict[str, Any],
    target: dict[str, Any],
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    change = target["change_from_previous"]
    return {
        "model": record["model"],
        "client_id": record["client_id"],
        "target_id": target["target_id"],
        "current_session_index": record["session_index"],
        "prior_session_index": (
            int(record["session_index"]) - 1 if prior is not None else None
        ),
        "comparison_scope": "immediately_previous_profile",
        "change_from_previous": change,
        "comparable": int(change in {"improved", "stable", "worsened"}),
        "better": int(change == "improved"),
        "worse": int(change == "worsened"),
        "improved": int(change == "improved"),
        "stable": int(change == "stable"),
        "worsened": int(change == "worsened"),
        "not_assessed": int(change == "not_assessed"),
        "carried_forward": target["carried_forward"],
        "newly_added": target["newly_added"],
        "current_performance_observed": target["performance_observed"],
        "prior_performance_observed": (
            prior["performance_observed"] if prior is not None else None
        ),
        "current_evidence_source_session": target["evidence_source_session"],
        "current_verbatim_evidence": _json_text(target["verbatim_evidence"]),
        "prior_verbatim_evidence": (
            _json_text(prior["verbatim_evidence"]) if prior is not None else None
        ),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("client_id cannot be converted to a safe registry filename")
    return cleaned
