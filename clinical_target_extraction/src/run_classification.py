"""Batched, independent client-level pre-treatment classification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .classification_store import ClassificationStore
from .data import ClientNote
from .model_client import ContextOverflowError, ModelConfig, VLLMClient, repair_prompt
from .prompts import build_classification_prompt, load_prompts, prompt_bundle_hash
from .schemas import ClassificationResultOutput, schema_for_prompt_style


ARCHITECTURE_VERSION = "pretreatment_classification_v1"


@dataclass(frozen=True)
class ClassificationWork:
    client: ClientNote
    task_prompt: str
    fingerprint: str


@dataclass
class ClassificationResult:
    output: ClassificationResultOutput | None = None
    input_tokens: int = 0
    error: Exception | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_clients(
    *,
    clients: list[ClientNote],
    selected_clients: list[str],
    model_config: ModelConfig,
    output_root: Path,
    batch_size: int = 8,
    prompt_style: str = "standard",
) -> dict[str, int]:
    """Classify each selected client once; no client depends on any other row."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if model_config.max_num_seqs != batch_size:
        model_config = model_config.model_copy(update={"max_num_seqs": batch_size})

    by_id = {client.client_id: client for client in clients}
    missing = [client_id for client_id in selected_clients if client_id not in by_id]
    if missing:
        raise ValueError(f"Selected client IDs are absent from the validated input: {missing}")

    prompts = load_prompts(prompt_style)
    prompts_hash = prompt_bundle_hash(prompts)
    output_schema = schema_for_prompt_style(prompt_style)
    output_label = (
        model_config.model_label
        if prompt_style == "standard"
        else f"{model_config.model_label}_brief_reasoning"
    )
    output_dir = output_root / output_label
    store = ClassificationStore(output_dir)
    latest = store.latest_validated()
    store.write_run_config(
        {
            "created_at": utc_now(),
            "architecture": ARCHITECTURE_VERSION,
            "selected_clients": selected_clients,
            "model": model_config.model_dump(),
            "prompt_bundle_sha256": prompts_hash,
            "prompt_style": prompt_style,
            "output_schema": output_schema.__name__,
            "output_directory": str(output_dir.resolve()),
            "inference_engine": "vllm.LLM.generate",
            "batch_size": batch_size,
            "input_unit": "one_combined_note_per_client",
            "combined_fields": ["SP text", "assessment text"],
        }
    )

    counters = {"completed": 0, "reused": 0, "failed": 0, "context_overflow": 0}
    pending: list[ClassificationWork] = []
    for client_id in selected_clients:
        client = by_id[client_id]
        task_prompt = build_classification_prompt(prompts["classification"], client)
        fingerprint = _request_fingerprint(
            client=client,
            prompts_hash=prompts_hash,
            model_config=model_config,
            prompt_style=prompt_style,
        )
        existing = latest.get(client_id)
        if existing and existing.get("request_fingerprint") == fingerprint:
            counters["reused"] += 1
            continue
        pending.append(ClassificationWork(client, task_prompt, fingerprint))

    print(
        f">>> Independent client classification with max_num_seqs={batch_size}; "
        f"prompt_style={prompt_style}; each request contains one combined "
        "pre-treatment note",
        flush=True,
    )
    progress = tqdm(
        total=len(selected_clients),
        initial=counters["reused"],
        desc=f"Classifying with {model_config.model_label}",
        unit="client",
        dynamic_ncols=True,
    )
    model_client: VLLMClient | None = None
    try:
        if pending:
            model_client = VLLMClient(model_config)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            assert model_client is not None
            results = _classify_batch(
                model_client=model_client,
                store=store,
                system_prompt=prompts["system"],
                works=batch,
                output_schema=output_schema,
            )
            for work, result in zip(batch, results, strict=True):
                client_id = work.client.client_id
                if result.error is not None:
                    is_overflow = isinstance(result.error, ContextOverflowError)
                    store.append_failure(
                        {
                            "architecture": ARCHITECTURE_VERSION,
                            "model": model_config.model_label,
                            "prompt_style": prompt_style,
                            "client_id": client_id,
                            "source_row": work.client.source_row,
                            "created_at": utc_now(),
                            "reason": (
                                "context_overflow" if is_overflow else "classification_failed"
                            ),
                            "error": str(result.error),
                        }
                    )
                    counters["context_overflow" if is_overflow else "failed"] += 1
                    progress.set_postfix_str(f"{client_id} failed", refresh=False)
                    progress.update(1)
                    continue

                assert result.output is not None
                store.append_validated(
                    {
                        "architecture": ARCHITECTURE_VERSION,
                        "model": model_config.model_label,
                        "client_id": client_id,
                        "source_row": work.client.source_row,
                        "treatment_category": result.output.treatment_category,
                        "brief_reasoning": getattr(
                            result.output,
                            "brief_reasoning",
                            None,
                        ),
                        "prompt_style": prompt_style,
                        "completed_at": utc_now(),
                        "request_fingerprint": work.fingerprint,
                        "input_sha256": _sha256_text(work.client.note_text),
                        "sp_text_sha256": _sha256_text(work.client.sp_text),
                        "assessment_text_sha256": _sha256_text(
                            work.client.assessment_text
                        ),
                        "prompt_bundle_sha256": prompts_hash,
                        "input_tokens": result.input_tokens,
                        "model_output": result.output.model_dump(),
                    }
                )
                counters["completed"] += 1
                progress.set_postfix_str(client_id, refresh=False)
                progress.update(1)
    finally:
        progress.close()
        try:
            store.materialize(selected_clients)
        finally:
            if model_client is not None:
                model_client.close()
    return counters


def _classify_batch(
    *,
    model_client: VLLMClient,
    store: ClassificationStore,
    system_prompt: str,
    works: list[ClassificationWork],
    output_schema: type[BaseModel],
) -> list[ClassificationResult]:
    requests = [
        (system_prompt, work.task_prompt, output_schema, output_schema.__name__)
        for work in works
    ]
    results = [ClassificationResult() for _ in works]
    try:
        completions = model_client.complete_batch(requests)
    except Exception as exc:
        return [ClassificationResult(error=exc) for _ in works]

    repair_indices: list[int] = []
    repair_requests: list[tuple[str, str, type[BaseModel], str]] = []
    for index, (work, completion) in enumerate(zip(works, completions, strict=True)):
        if completion.error is not None:
            results[index].error = completion.error
            continue
        raw = completion.text or ""
        try:
            output = output_schema.model_validate_json(raw)
            _validate_client_id(output, work.client.client_id)
        except (ValidationError, ValueError) as exc:
            messages = _validation_messages(exc)
            _record_raw(
                store, work, "initial", completion.input_tokens, False, messages, raw
            )
            repair_indices.append(index)
            repair_requests.append(
                (
                    system_prompt,
                    repair_prompt(
                        work.task_prompt,
                        raw,
                        messages,
                        output_schema,
                    ),
                    output_schema,
                    f"{output_schema.__name__}Repair",
                )
            )
            continue
        _record_raw(store, work, "initial", completion.input_tokens, True, [], raw)
        results[index] = ClassificationResult(output, completion.input_tokens)

    if not repair_requests:
        return results
    try:
        repairs = model_client.complete_batch(repair_requests)
    except Exception as exc:
        for index in repair_indices:
            results[index].error = exc
        return results

    for index, completion in zip(repair_indices, repairs, strict=True):
        work = works[index]
        if completion.error is not None:
            results[index].error = completion.error
            continue
        raw = completion.text or ""
        try:
            output = output_schema.model_validate_json(raw)
            _validate_client_id(output, work.client.client_id)
        except (ValidationError, ValueError) as exc:
            messages = _validation_messages(exc)
            _record_raw(
                store, work, "repair", completion.input_tokens, False, messages, raw
            )
            results[index].error = RuntimeError(
                "Model output remained invalid after one repair attempt: "
                + "; ".join(messages)
            )
            continue
        _record_raw(store, work, "repair", completion.input_tokens, True, [], raw)
        results[index] = ClassificationResult(output, completion.input_tokens)
    return results


def _validate_client_id(output: BaseModel, expected: str) -> None:
    client_id = getattr(output, "client_id", None)
    if client_id != expected:
        raise ValueError(
            f"client_id must equal the requested ID {expected!r}; got {client_id!r}"
        )


def _validation_messages(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors()
        ]
    return [str(exc)]


def _record_raw(
    store: ClassificationStore,
    work: ClassificationWork,
    attempt: str,
    input_tokens: int,
    valid: bool,
    errors: list[str],
    raw: str,
) -> None:
    store.append_raw(
        {
            "architecture": ARCHITECTURE_VERSION,
            "client_id": work.client.client_id,
            "source_row": work.client.source_row,
            "attempt": attempt,
            "created_at": utc_now(),
            "input_tokens": input_tokens,
            "valid": valid,
            "validation_errors": errors,
            "raw_response": raw,
        }
    )


def _request_fingerprint(
    *,
    client: ClientNote,
    prompts_hash: str,
    model_config: ModelConfig,
    prompt_style: str = "standard",
) -> str:
    payload: dict[str, Any] = {
        "architecture": ARCHITECTURE_VERSION,
        "client_id": client.client_id,
        "sp_text": client.sp_text,
        "assessment_text": client.assessment_text,
        "prompt_bundle_sha256": prompts_hash,
        "prompt_style": prompt_style,
        "model_config": model_config.model_dump(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
