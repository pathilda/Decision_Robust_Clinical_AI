"""Rolling-profile clinical target extraction orchestration."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .data import ClinicalNote
from .model_client import ContextOverflowError, ModelConfig, VLLMClient, repair_prompt
from .output_store import OutputStore
from .prompts import build_later_prompt, build_session1_prompt, load_prompts, prompt_bundle_hash
from .registry import TargetRegistry
from .schemas import (
    LaterSessionOutput,
    ProfileTarget,
    Session1Output,
    SessionProfile,
)
from .validate import (
    OutputValidationError,
    audit_later_session_quality,
    audit_session1_quality,
    validate_later_session,
    validate_session1,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)
ARCHITECTURE_VERSION = "rolling_snapshot_v1"


class SessionFailed(RuntimeError):
    pass


@dataclass
class StructuredCall:
    system_prompt: str
    task_prompt: str
    schema: type[BaseModel]
    schema_name: str
    validate_context: Callable[[Any], None]
    audit_context: dict[str, Any]


@dataclass
class StructuredResult:
    output: BaseModel | None = None
    input_tokens: int = 0
    error: Exception | None = None


@dataclass
class ClientState:
    client_id: str
    notes: list[ClinicalNote]
    latest: dict[int, dict[str, Any]]
    registry: TargetRegistry = field(init=False)
    previous_profile: SessionProfile | None = None
    position: int = 0
    stopped: bool = False
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "completed": 0,
            "reused": 0,
            "failed": 0,
            "context_overflow": 0,
        }
    )

    def __post_init__(self) -> None:
        self.registry = TargetRegistry(client_id=self.client_id)


@dataclass
class SessionWork:
    state: ClientState
    note: ClinicalNote
    registry_before: TargetRegistry
    previous_profile: SessionProfile | None
    fingerprint: str
    audit_context: dict[str, Any]
    output: Session1Output | LaterSessionOutput | None = None
    session_profile: SessionProfile | None = None
    input_tokens: int = 0
    quality_warnings: list[dict[str, str]] = field(default_factory=list)
    target_assignments: list[dict[str, Any]] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validation_messages(exc: Exception) -> list[str]:
    if isinstance(exc, OutputValidationError):
        return exc.errors
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors()
        ]
    return [str(exc)]


def call_with_one_repair(
    *,
    model_client: VLLMClient,
    store: OutputStore,
    system_prompt: str,
    task_prompt: str,
    schema: type[SchemaT],
    schema_name: str,
    validate_context: Callable[[SchemaT], None],
    audit_context: dict[str, Any],
) -> tuple[SchemaT, int]:
    """Single-call compatibility wrapper used by tests and small integrations."""

    raw, input_tokens = model_client.complete(system_prompt, task_prompt, schema, schema_name)
    try:
        parsed = schema.model_validate_json(raw)
        validate_context(parsed)
    except (ValidationError, OutputValidationError, ValueError) as first_error:
        messages = validation_messages(first_error)
        _record_raw(store, audit_context, "initial", input_tokens, False, messages, raw)
        repaired_raw, repair_tokens = model_client.complete(
            system_prompt,
            repair_prompt(task_prompt, raw, messages, schema),
            schema,
            f"{schema_name}Repair",
        )
        try:
            parsed = schema.model_validate_json(repaired_raw)
            validate_context(parsed)
        except (ValidationError, OutputValidationError, ValueError) as second_error:
            second_messages = validation_messages(second_error)
            _record_raw(
                store,
                audit_context,
                "repair",
                repair_tokens,
                False,
                second_messages,
                repaired_raw,
            )
            raise SessionFailed(
                "Model output remained invalid after one repair attempt: "
                + "; ".join(second_messages)
            ) from second_error
        _record_raw(store, audit_context, "repair", repair_tokens, True, [], repaired_raw)
        return parsed, repair_tokens

    _record_raw(store, audit_context, "initial", input_tokens, True, [], raw)
    return parsed, input_tokens


def call_batch_with_one_repair(
    *,
    model_client: VLLMClient,
    store: OutputStore,
    calls: list[StructuredCall],
) -> list[StructuredResult]:
    """Run initial requests and only structurally invalid repairs as two batches."""

    if not calls:
        return []
    requests = [
        (call.system_prompt, call.task_prompt, call.schema, call.schema_name)
        for call in calls
    ]
    results = [StructuredResult() for _ in calls]
    try:
        completions = model_client.complete_batch(requests)
    except Exception as exc:
        return [StructuredResult(error=exc) for _ in calls]

    repair_calls: list[StructuredCall] = []
    repair_indices: list[int] = []
    for index, (call, completion) in enumerate(zip(calls, completions, strict=True)):
        if completion.error is not None:
            results[index].error = completion.error
            continue
        raw = completion.text or ""
        try:
            parsed = call.schema.model_validate_json(raw)
            call.validate_context(parsed)
        except (ValidationError, OutputValidationError, ValueError) as error:
            messages = validation_messages(error)
            _record_raw(
                store,
                call.audit_context,
                "initial",
                completion.input_tokens,
                False,
                messages,
                raw,
            )
            repair_calls.append(
                StructuredCall(
                    system_prompt=call.system_prompt,
                    task_prompt=repair_prompt(call.task_prompt, raw, messages, call.schema),
                    schema=call.schema,
                    schema_name=f"{call.schema_name}Repair",
                    validate_context=call.validate_context,
                    audit_context=call.audit_context,
                )
            )
            repair_indices.append(index)
            continue
        _record_raw(
            store,
            call.audit_context,
            "initial",
            completion.input_tokens,
            True,
            [],
            raw,
        )
        results[index] = StructuredResult(parsed, completion.input_tokens)

    if not repair_calls:
        return results

    repair_requests = [
        (call.system_prompt, call.task_prompt, call.schema, call.schema_name)
        for call in repair_calls
    ]
    try:
        repairs = model_client.complete_batch(repair_requests)
    except Exception as exc:
        for result_index in repair_indices:
            results[result_index].error = exc
        return results

    for result_index, call, completion in zip(
        repair_indices,
        repair_calls,
        repairs,
        strict=True,
    ):
        if completion.error is not None:
            results[result_index].error = completion.error
            continue
        raw = completion.text or ""
        try:
            parsed = call.schema.model_validate_json(raw)
            call.validate_context(parsed)
        except (ValidationError, OutputValidationError, ValueError) as error:
            messages = validation_messages(error)
            _record_raw(
                store,
                call.audit_context,
                "repair",
                completion.input_tokens,
                False,
                messages,
                raw,
            )
            results[result_index].error = SessionFailed(
                "Model output remained invalid after one repair attempt: "
                + "; ".join(messages)
            )
            continue
        _record_raw(
            store,
            call.audit_context,
            "repair",
            completion.input_tokens,
            True,
            [],
            raw,
        )
        results[result_index] = StructuredResult(parsed, completion.input_tokens)
    return results


def _record_raw(
    store: OutputStore,
    audit_context: dict[str, Any],
    attempt: str,
    input_tokens: int,
    valid: bool,
    errors: list[str],
    raw: str,
) -> None:
    store.append_raw(
        {
            **audit_context,
            "attempt": attempt,
            "created_at": utc_now(),
            "input_tokens": input_tokens,
            "valid": valid,
            "validation_errors": errors,
            "raw_response": raw,
        }
    )


def request_fingerprint(
    *,
    current_note: ClinicalNote,
    previous_profile: SessionProfile | None,
    registry: TargetRegistry,
    prompts_hash: str,
    model_config: ModelConfig,
) -> str:
    """Hash only the current note and immediately previous compact state."""

    return sha256_json(
        {
            "architecture": ARCHITECTURE_VERSION,
            "current_note": current_note.__dict__,
            "previous_profile": (
                previous_profile.model_dump() if previous_profile is not None else None
            ),
            "registry_before": registry.model_dump(),
            "prompt_bundle_sha256": prompts_hash,
            "model_config": model_config.model_dump(),
        }
    )


def process_clients(
    *,
    notes: list[ClinicalNote],
    selected_clients: list[str],
    model_config: ModelConfig,
    output_root: Path,
    batch_size: int = 8,
) -> dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if model_config.max_num_seqs != batch_size:
        model_config = model_config.model_copy(update={"max_num_seqs": batch_size})

    prompts = load_prompts()
    prompts_hash = prompt_bundle_hash(prompts)
    output_dir = output_root / model_config.model_label
    store = OutputStore(output_dir)
    latest = store.latest_validated()
    grouped: dict[str, list[ClinicalNote]] = defaultdict(list)
    for note in notes:
        grouped[note.client_id].append(note)

    missing = [client_id for client_id in selected_clients if client_id not in grouped]
    if missing:
        raise ValueError(f"Selected client IDs are absent from the validated input: {missing}")

    store.write_run_config(
        {
            "created_at": utc_now(),
            "architecture": ARCHITECTURE_VERSION,
            "selected_clients": selected_clients,
            "model": model_config.model_dump(),
            "prompt_bundle_sha256": prompts_hash,
            "output_directory": str(output_dir.resolve()),
            "inference_engine": "vllm.LLM.generate",
            "batch_size": batch_size,
            "history_policy": "previous_session_profile_plus_current_note",
            "canonicalization_calls": 0,
            "table_materialization": "end_of_run",
        }
    )

    model_client = VLLMClient(model_config)
    selected_note_count = sum(len(grouped[client_id]) for client_id in selected_clients)
    print(
        f">>> Rolling-profile generation with max_num_seqs={batch_size}; "
        "each prompt contains only the previous JSON profile and current note",
        flush=True,
    )
    progress = tqdm(
        total=selected_note_count,
        desc=f"Extracting with {model_config.model_label}",
        unit="session",
        dynamic_ncols=True,
    )
    latest_by_client = {
        client_id: {
            session_index: record
            for (record_client, session_index), record in latest.items()
            if record_client == client_id
        }
        for client_id in selected_clients
    }
    states = [
        ClientState(client_id, grouped[client_id], latest_by_client[client_id])
        for client_id in selected_clients
    ]

    try:
        while True:
            work_items: list[SessionWork] = []
            for state in states:
                if state.stopped:
                    continue
                work = _next_session_work(
                    state=state,
                    model_config=model_config,
                    store=store,
                    prompts_hash=prompts_hash,
                    progress=progress,
                )
                if work is not None:
                    work_items.append(work)
            if not work_items:
                break

            results = call_batch_with_one_repair(
                model_client=model_client,
                store=store,
                calls=[_session_call(work, prompts) for work in work_items],
            )
            for work, result in zip(work_items, results, strict=True):
                if result.error is not None:
                    _fail_work(work, result.error, store, progress)
                    continue
                work.input_tokens = result.input_tokens
                output = result.output
                if isinstance(output, Session1Output):
                    work.output = output
                    work.quality_warnings = audit_session1_quality(output, work.note)
                    work.session_profile = _initial_profile(work, output)
                elif isinstance(output, LaterSessionOutput):
                    work.output = output
                    assert work.previous_profile is not None
                    work.quality_warnings = audit_later_session_quality(
                        output,
                        work.note,
                        work.previous_profile,
                    )
                    work.session_profile = _updated_profile(work, output)
                else:
                    _fail_work(
                        work,
                        RuntimeError("Unexpected session output type"),
                        store,
                        progress,
                    )
                    continue
                _commit_work(work, prompts_hash, store, progress)
    finally:
        progress.close()
        try:
            store.materialize_tables()
        finally:
            model_client.close()

    counters = {"completed": 0, "reused": 0, "failed": 0, "context_overflow": 0}
    for state in states:
        for name, value in state.counters.items():
            counters[name] += value
    return counters


def _next_session_work(
    *,
    state: ClientState,
    model_config: ModelConfig,
    store: OutputStore,
    prompts_hash: str,
    progress: Any,
) -> SessionWork | None:
    while state.position < len(state.notes):
        note = state.notes[state.position]
        registry_before = state.registry.model_copy(deep=True)
        previous_profile = (
            state.previous_profile.model_copy(deep=True)
            if state.previous_profile is not None
            else None
        )
        fingerprint = request_fingerprint(
            current_note=note,
            previous_profile=previous_profile,
            registry=state.registry,
            prompts_hash=prompts_hash,
            model_config=model_config,
        )
        existing = state.latest.get(note.session_index)
        if existing and existing.get("request_fingerprint") == fingerprint:
            try:
                resumed_registry = TargetRegistry.model_validate(existing["registry_after"])
                resumed_profile = SessionProfile.model_validate(existing["session_profile"])
            except (KeyError, ValidationError):
                existing = None
            else:
                state.registry = resumed_registry
                state.previous_profile = resumed_profile
                store.save_registry(state.client_id, state.registry.model_dump())
                state.counters["reused"] += 1
                state.position += 1
                progress.set_postfix_str(
                    f"{state.client_id} S{note.session_index} reused",
                    refresh=False,
                )
                progress.update(1)
                continue

        stale_sessions = [
            session_index
            for session_index in state.latest
            if session_index >= note.session_index
        ]
        if stale_sessions:
            store.invalidate_from(
                state.client_id,
                note.session_index,
                "Rolling-profile fingerprint or upstream checkpoint changed",
            )
            for session_index in stale_sessions:
                del state.latest[session_index]
            store.save_registry(state.client_id, state.registry.model_dump())

        return SessionWork(
            state=state,
            note=note,
            registry_before=registry_before,
            previous_profile=previous_profile,
            fingerprint=fingerprint,
            audit_context={
                "model": model_config.model_label,
                "client_id": state.client_id,
                "session_index": note.session_index,
                "note_id": note.note_id,
                "call_type": "session_profile_update",
            },
        )
    return None


def _session_call(work: SessionWork, prompts: dict[str, str]) -> StructuredCall:
    if work.note.session_index == 1:
        return StructuredCall(
            system_prompt=prompts["system"],
            task_prompt=build_session1_prompt(prompts["session1"], work.note),
            schema=Session1Output,
            schema_name="Session1Output",
            validate_context=lambda value: validate_session1(value, work.note),
            audit_context=work.audit_context,
        )

    previous_profile = work.previous_profile
    if previous_profile is None:
        raise RuntimeError("A later session requires the immediately previous profile")
    return StructuredCall(
        system_prompt=prompts["system"],
        task_prompt=build_later_prompt(
            prompts["later_session"],
            previous_profile.model_dump(),
            work.note,
        ),
        schema=LaterSessionOutput,
        schema_name="LaterSessionOutput",
        validate_context=lambda value: validate_later_session(
            value,
            work.note,
            previous_profile,
        ),
        audit_context=work.audit_context,
    )


def _initial_profile(work: SessionWork, output: Session1Output) -> SessionProfile:
    targets: list[ProfileTarget] = []
    for index, discovered in enumerate(output.targets):
        target_id = work.state.registry.add_discovered(discovered, session_index=1)
        registered = work.state.registry.get(target_id)
        work.target_assignments.append(
            {
                "source": "initial_discovery",
                "source_index": index,
                "proposed_label": discovered.proposed_label,
                "target_id": target_id,
            }
        )
        targets.append(
            ProfileTarget(
                target_id=target_id,
                canonical_label=registered.canonical_label,
                definition=registered.definition,
                aliases=registered.aliases,
                verbatim_evidence=list(discovered.verbatim_evidence),
                evidence_source_session=1,
                substantively_treated=discovered.substantively_treated,
                performance_observed=discovered.performance_observed,
                change_from_previous="new",
                carried_forward=0,
                newly_added=1,
            )
        )
    return SessionProfile(
        client_id=output.client_id,
        session_index=1,
        note_id=output.note_id,
        targets=targets,
    )


def _updated_profile(work: SessionWork, output: LaterSessionOutput) -> SessionProfile:
    by_id = {target.target_id: target for target in output.existing_targets}
    targets: list[ProfileTarget] = []

    for registered in work.registry_before.targets:
        update = by_id[registered.target_id]
        targets.append(
            ProfileTarget(
                target_id=registered.target_id,
                canonical_label=registered.canonical_label,
                definition=registered.definition,
                aliases=registered.aliases,
                verbatim_evidence=list(update.verbatim_evidence),
                evidence_source_session=update.evidence_source_session,
                substantively_treated=update.substantively_treated,
                performance_observed=update.performance_observed,
                change_from_previous=update.change_from_previous,
                carried_forward=update.carried_forward,
                newly_added=0,
            )
        )

    for index, candidate in enumerate(output.new_targets):
        target_id = work.state.registry.add_new(candidate, work.note.session_index)
        registered = work.state.registry.get(target_id)
        work.target_assignments.append(
            {
                "source": "new_target_discovery",
                "source_index": index,
                "proposed_label": candidate.proposed_label,
                "target_id": target_id,
            }
        )
        targets.append(
            ProfileTarget(
                target_id=target_id,
                canonical_label=registered.canonical_label,
                definition=registered.definition,
                aliases=registered.aliases,
                verbatim_evidence=list(candidate.verbatim_evidence),
                evidence_source_session=work.note.session_index,
                substantively_treated=candidate.substantively_treated,
                performance_observed=candidate.performance_observed,
                change_from_previous="new",
                carried_forward=0,
                newly_added=1,
            )
        )

    return SessionProfile(
        client_id=output.client_id,
        session_index=output.session_index,
        note_id=output.note_id,
        targets=targets,
    )


def _commit_work(
    work: SessionWork,
    prompts_hash: str,
    store: OutputStore,
    progress: Any,
) -> None:
    output = work.output
    session_profile = work.session_profile
    assert output is not None and session_profile is not None
    envelope = {
        **{key: value for key, value in work.audit_context.items() if key != "call_type"},
        "architecture": ARCHITECTURE_VERSION,
        "completed_at": utc_now(),
        "request_fingerprint": work.fingerprint,
        "input_note_sha256": sha256_text(work.note.note_text),
        "prompt_bundle_sha256": prompts_hash,
        "input_tokens": work.input_tokens,
        "quality_warnings": work.quality_warnings,
        "model_output": output.model_dump(),
        "target_assignments": work.target_assignments,
        "registry_before": work.registry_before.model_dump(),
        "registry_after": work.state.registry.model_dump(),
        "session_profile": session_profile.model_dump(),
    }
    store.append_validated(envelope)
    store.save_registry(work.state.client_id, work.state.registry.model_dump())
    work.state.latest[work.note.session_index] = envelope
    work.state.previous_profile = session_profile
    work.state.counters["completed"] += 1
    work.state.position += 1
    progress.set_postfix_str(
        f"{work.state.client_id} S{work.note.session_index}",
        refresh=False,
    )
    progress.update(1)


def _fail_work(
    work: SessionWork,
    error: Exception,
    store: OutputStore,
    progress: Any,
) -> None:
    is_overflow = isinstance(error, ContextOverflowError)
    store.append_failure(
        {
            **{key: value for key, value in work.audit_context.items() if key != "call_type"},
            "architecture": ARCHITECTURE_VERSION,
            "created_at": utc_now(),
            "reason": "context_overflow" if is_overflow else "session_failed",
            "error": str(error),
        }
    )
    counter = "context_overflow" if is_overflow else "failed"
    work.state.counters[counter] += 1
    work.state.stopped = True
    progress.set_postfix_str(
        f"{work.state.client_id} S{work.note.session_index} failed",
        refresh=False,
    )
    progress.update(1)
