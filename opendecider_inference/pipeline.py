"""Resumable local OpenDecider classification over validated client notes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tqdm.auto import tqdm

from clinical_target_extraction.src.data import ClientNote

from .local_model import LocalOpenDecider
from .rubric import (
    CATEGORIES,
    CATEGORY_CRITERIA,
    MODEL_LABEL,
    QUESTION_INSTRUCTIONS,
    QUESTION_NAME,
    decision_questions,
    rubric_hash,
)
from .store import OpenDeciderStore


ARCHITECTURE_VERSION = "opendecider_pretreatment_classification_v1"


class DecisionModel(Protocol):
    def classify_batch(
        self,
        states: list[str],
        questions: dict[str, dict[str, object]],
    ) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


ModelFactory = Callable[..., DecisionModel]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_clients(
    *,
    clients: list[ClientNote],
    selected_clients: list[str],
    adapter_path: Path,
    base_path: Path,
    output_root: Path,
    batch_size: int = 8,
    device: str = "cuda",
    model_factory: ModelFactory = LocalOpenDecider,
) -> dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    adapter_path = adapter_path.expanduser().resolve()
    base_path = base_path.expanduser().resolve()
    by_id = {client.client_id: client for client in clients}
    missing = [client_id for client_id in selected_clients if client_id not in by_id]
    if missing:
        raise ValueError(f"Selected client IDs are absent from the validated input: {missing}")

    adapter_config_hash = _required_file_hash(adapter_path / "adapter_config.json")
    adapter_metadata_hash = _required_file_hash(adapter_path / "opendecider.json")
    base_config_hash = _required_file_hash(base_path / "config.json")
    current_rubric_hash = rubric_hash()
    output_dir = output_root.expanduser().resolve() / MODEL_LABEL
    store = OpenDeciderStore(output_dir)
    latest = store.latest_validated()
    store.write_run_config(
        {
            "created_at": utc_now(),
            "architecture": ARCHITECTURE_VERSION,
            "selected_clients": selected_clients,
            "model": MODEL_LABEL,
            "adapter_path": str(adapter_path),
            "base_path": str(base_path),
            "adapter_config_sha256": adapter_config_hash,
            "adapter_metadata_sha256": adapter_metadata_hash,
            "base_config_sha256": base_config_hash,
            "rubric_sha256": current_rubric_hash,
            "question": QUESTION_INSTRUCTIONS,
            "category_criteria": CATEGORY_CRITERIA,
            "device": device,
            "batch_size": batch_size,
            "local_only": True,
            "input_unit": "one_combined_note_per_client",
            "combined_fields": ["SP text", "assessment text"],
        }
    )

    counters = {"completed": 0, "reused": 0, "failed": 0}
    pending: list[tuple[ClientNote, str]] = []
    for client_id in selected_clients:
        client = by_id[client_id]
        fingerprint = _request_fingerprint(
            client=client,
            adapter_config_hash=adapter_config_hash,
            adapter_metadata_hash=adapter_metadata_hash,
            base_config_hash=base_config_hash,
            current_rubric_hash=current_rubric_hash,
        )
        existing = latest.get(client_id)
        if existing and existing.get("request_fingerprint") == fingerprint:
            counters["reused"] += 1
        else:
            pending.append((client, fingerprint))

    print(
        f">>> Local OpenDecider classification on {device}; "
        f"batch_size={batch_size}, no hosted inference",
        flush=True,
    )
    progress = tqdm(
        total=len(selected_clients),
        initial=counters["reused"],
        desc="Classifying with OpenDecider",
        unit="client",
        dynamic_ncols=True,
    )
    model: DecisionModel | None = None
    try:
        if pending:
            model = model_factory(adapter_path, base_path, device=device)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            assert model is not None
            batch_results = _infer_with_isolation(model, batch)
            for (client, fingerprint), result in zip(
                batch,
                batch_results,
                strict=True,
            ):
                if isinstance(result, Exception):
                    _record_failure(store, client, result)
                    counters["failed"] += 1
                    progress.set_postfix_str(f"{client.client_id} failed", refresh=False)
                    progress.update(1)
                    continue
                try:
                    normalized = validate_response(result)
                except (KeyError, TypeError, ValueError) as exc:
                    store.append_response(
                        {
                            "architecture": ARCHITECTURE_VERSION,
                            "client_id": client.client_id,
                            "source_row": client.source_row,
                            "created_at": utc_now(),
                            "valid": False,
                            "validation_error": str(exc),
                            "response": result,
                        }
                    )
                    _record_failure(store, client, exc)
                    counters["failed"] += 1
                    progress.set_postfix_str(f"{client.client_id} failed", refresh=False)
                    progress.update(1)
                    continue

                store.append_response(
                    {
                        "architecture": ARCHITECTURE_VERSION,
                        "client_id": client.client_id,
                        "source_row": client.source_row,
                        "created_at": utc_now(),
                        "valid": True,
                        "response": result,
                    }
                )
                completed_at = utc_now()
                store.append_validated(
                    {
                        "architecture": ARCHITECTURE_VERSION,
                        "model": MODEL_LABEL,
                        "client_id": client.client_id,
                        "source_row": client.source_row,
                        "treatment_category": normalized["treatment_category"],
                        "confidence": normalized["confidence"],
                        "probabilities": normalized["probabilities"],
                        "completed_at": completed_at,
                        "request_fingerprint": fingerprint,
                        "input_sha256": _sha256_text(client.note_text),
                        "sp_text_sha256": _sha256_text(client.sp_text),
                        "assessment_text_sha256": _sha256_text(
                            client.assessment_text
                        ),
                        "rubric_sha256": current_rubric_hash,
                        "input_tokens": normalized["input_tokens"],
                    }
                )
                counters["completed"] += 1
                progress.set_postfix_str(client.client_id, refresh=False)
                progress.update(1)
    finally:
        progress.close()
        try:
            store.materialize(selected_clients)
        finally:
            if model is not None:
                model.close()
    return counters


def _infer_with_isolation(
    model: DecisionModel,
    batch: list[tuple[ClientNote, str]],
) -> list[dict[str, Any] | Exception]:
    states = [client.note_text for client, _ in batch]
    questions = decision_questions()
    try:
        responses = model.classify_batch(states, questions)
        if len(responses) != len(batch):
            raise RuntimeError(
                f"OpenDecider returned {len(responses)} responses for {len(batch)} clients"
            )
        return responses
    except Exception as batch_error:
        if len(batch) == 1:
            return [batch_error]

    isolated: list[dict[str, Any] | Exception] = []
    for client, _ in batch:
        try:
            responses = model.classify_batch([client.note_text], questions)
            if len(responses) != 1:
                raise RuntimeError(
                    f"OpenDecider returned {len(responses)} responses for one client"
                )
            isolated.append(responses[0])
        except Exception as exc:
            isolated.append(exc)
    return isolated


def validate_response(response: dict[str, Any]) -> dict[str, Any]:
    answer = response["answers"][QUESTION_NAME]
    choice = str(answer["choice"])
    if choice not in CATEGORIES:
        raise ValueError(f"Unsupported treatment category: {choice!r}")
    raw_probabilities = answer["probabilities"]
    if set(raw_probabilities) != set(CATEGORIES):
        raise ValueError(
            "Probability keys must exactly match the three treatment categories"
        )
    probabilities = {
        category: float(raw_probabilities[category]) for category in CATEGORIES
    }
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Every category probability must be a finite value from 0 to 1")
    probability_sum = sum(probabilities.values())
    if not math.isclose(probability_sum, 1.0, abs_tol=0.02):
        raise ValueError(f"Category probabilities sum to {probability_sum}, not 1")
    confidence = float(answer.get("confidence", probabilities[choice]))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a finite value from 0 to 1")
    input_tokens = int(response.get("usage", {}).get("input_tokens", 0))
    if input_tokens < 0:
        raise ValueError("input_tokens cannot be negative")
    return {
        "treatment_category": choice,
        "confidence": confidence,
        "probabilities": probabilities,
        "input_tokens": input_tokens,
    }


def _record_failure(
    store: OpenDeciderStore,
    client: ClientNote,
    error: Exception,
) -> None:
    store.append_failure(
        {
            "architecture": ARCHITECTURE_VERSION,
            "model": MODEL_LABEL,
            "client_id": client.client_id,
            "source_row": client.source_row,
            "created_at": utc_now(),
            "reason": "classification_failed",
            "error": str(error),
        }
    )


def _request_fingerprint(
    *,
    client: ClientNote,
    adapter_config_hash: str,
    adapter_metadata_hash: str,
    base_config_hash: str,
    current_rubric_hash: str,
) -> str:
    payload = {
        "architecture": ARCHITECTURE_VERSION,
        "client_id": client.client_id,
        "sp_text": client.sp_text,
        "assessment_text": client.assessment_text,
        "adapter_config_sha256": adapter_config_hash,
        "adapter_metadata_sha256": adapter_metadata_hash,
        "base_config_sha256": base_config_hash,
        "rubric_sha256": current_rubric_hash,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required local model file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
