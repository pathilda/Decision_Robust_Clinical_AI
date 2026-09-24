"""CLI and sequential per-client extraction orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .data import ClinicalNote, inspect_input, load_data_config, load_notes
from .model_client import (
    ContextOverflowError,
    ModelConfig,
    VLLMClient,
    load_model_config,
    repair_prompt,
)
from .output_store import OutputStore
from .prompts import (
    build_canonicalization_prompt,
    build_later_prompt,
    build_session1_prompt,
    load_prompts,
    prompt_bundle_hash,
)
from .registry import TargetRegistry
from .schemas import CanonicalizationOutput, LaterSessionOutput, Session1Output
from .validate import (
    OutputValidationError,
    validate_canonicalization,
    validate_later_session,
    validate_session1,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class SessionFailed(RuntimeError):
    pass


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
    raw, input_tokens = model_client.complete(system_prompt, task_prompt, schema, schema_name)
    try:
        parsed = schema.model_validate_json(raw)
        validate_context(parsed)
    except (ValidationError, OutputValidationError, ValueError) as first_error:
        first_messages = validation_messages(first_error)
        store.append_raw(
            {
                **audit_context,
                "attempt": "initial",
                "created_at": utc_now(),
                "input_tokens": input_tokens,
                "valid": False,
                "validation_errors": first_messages,
                "raw_response": raw,
            }
        )
        repair = repair_prompt(task_prompt, raw, first_messages, schema)
        repaired_raw, repair_tokens = model_client.complete(
            system_prompt,
            repair,
            schema,
            f"{schema_name}Repair",
        )
        try:
            parsed = schema.model_validate_json(repaired_raw)
            validate_context(parsed)
        except (ValidationError, OutputValidationError, ValueError) as second_error:
            second_messages = validation_messages(second_error)
            store.append_raw(
                {
                    **audit_context,
                    "attempt": "repair",
                    "created_at": utc_now(),
                    "input_tokens": repair_tokens,
                    "valid": False,
                    "validation_errors": second_messages,
                    "raw_response": repaired_raw,
                }
            )
            raise SessionFailed(
                "Model output remained invalid after one repair attempt: "
                + "; ".join(second_messages)
            ) from second_error
        store.append_raw(
            {
                **audit_context,
                "attempt": "repair",
                "created_at": utc_now(),
                "input_tokens": repair_tokens,
                "valid": True,
                "validation_errors": [],
                "raw_response": repaired_raw,
            }
        )
        return parsed, repair_tokens

    store.append_raw(
        {
            **audit_context,
            "attempt": "initial",
            "created_at": utc_now(),
            "input_tokens": input_tokens,
            "valid": True,
            "validation_errors": [],
            "raw_response": raw,
        }
    )
    return parsed, input_tokens


def request_fingerprint(
    *,
    notes_to_date: list[ClinicalNote],
    registry: TargetRegistry,
    prior_records: list[dict[str, Any]],
    prompts_hash: str,
    model_config: ModelConfig,
) -> str:
    model_settings = model_config.model_dump(exclude={"api_key"})
    return sha256_json(
        {
            "notes": [note.__dict__ for note in notes_to_date],
            "registry_before": registry.model_dump(),
            "prior_records": prior_records,
            "prompt_bundle_sha256": prompts_hash,
            "model_config": model_settings,
        }
    )


def process_clients(
    *,
    notes: list[ClinicalNote],
    selected_clients: list[str],
    model_config: ModelConfig,
    output_root: Path,
) -> dict[str, int]:
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
    if len(selected_clients) > 5:
        raise ValueError("Dry-run safety limit is five clients; do not run the full dataset yet")

    store.write_run_config(
        {
            "created_at": utc_now(),
            "selected_clients": selected_clients,
            "model": model_config.model_dump(exclude={"api_key"}),
            "prompt_bundle_sha256": prompts_hash,
            "output_directory": str(output_dir.resolve()),
        }
    )
    model_client = VLLMClient(model_config)
    counters = {"completed": 0, "reused": 0, "failed": 0, "context_overflow": 0}

    for client_id in selected_clients:
        registry = TargetRegistry(client_id=client_id)
        prior_records: list[dict[str, Any]] = []
        prior_observed: dict[tuple[int, str], int] = {}
        client_notes = grouped[client_id]
        notes_by_session = {note.session_index: note for note in client_notes}

        for position, note in enumerate(client_notes):
            notes_to_date = client_notes[: position + 1]
            registry_before = registry.model_copy(deep=True)
            fingerprint = request_fingerprint(
                notes_to_date=notes_to_date,
                registry=registry,
                prior_records=prior_records,
                prompts_hash=prompts_hash,
                model_config=model_config,
            )
            existing = latest.get((client_id, note.session_index))
            if existing and existing.get("request_fingerprint") == fingerprint:
                registry = TargetRegistry.model_validate(existing["registry_after"])
                prior_record = existing["prior_structured_record"]
                prior_records.append(prior_record)
                _update_observed(prior_observed, prior_record)
                store.save_registry(client_id, registry.model_dump())
                counters["reused"] += 1
                continue

            stale_keys = [
                key for key in latest if key[0] == client_id and key[1] >= note.session_index
            ]
            if stale_keys:
                store.invalidate_from(
                    client_id,
                    note.session_index,
                    "Request fingerprint or upstream checkpoint changed",
                )
                for key in stale_keys:
                    del latest[key]
                store.save_registry(client_id, registry.model_dump())
                store.materialize_parquet()

            audit_context = {
                "model": model_config.model_label,
                "client_id": client_id,
                "session_index": note.session_index,
                "note_id": note.note_id,
            }
            try:
                if note.session_index == 1:
                    task_prompt = build_session1_prompt(prompts["session1"], note)
                    output, input_tokens = call_with_one_repair(
                        model_client=model_client,
                        store=store,
                        system_prompt=prompts["system"],
                        task_prompt=task_prompt,
                        schema=Session1Output,
                        schema_name="Session1Output",
                        validate_context=lambda value: validate_session1(value, note),
                        audit_context={**audit_context, "call_type": "session_extraction"},
                    )
                    assignments = []
                    for index, target in enumerate(output.targets):
                        target_id = registry.add_discovered(target, session_index=1)
                        assignments.append(
                            {
                                "source_index": index,
                                "proposed_label": target.proposed_label,
                                "target_id": target_id,
                            }
                        )
                    canonicalizations: list[dict[str, Any]] = []
                    candidate_assignments: list[dict[str, Any]] = []
                    prior_record = _session1_prior_record(output, assignments)
                else:
                    task_prompt = build_later_prompt(
                        prompts["later_session"], registry, notes_to_date, prior_records
                    )
                    output, input_tokens = call_with_one_repair(
                        model_client=model_client,
                        store=store,
                        system_prompt=prompts["system"],
                        task_prompt=task_prompt,
                        schema=LaterSessionOutput,
                        schema_name="LaterSessionOutput",
                        validate_context=lambda value: validate_later_session(
                            value,
                            note,
                            notes_by_session,
                            registry_before,
                            prior_observed,
                        ),
                        audit_context={**audit_context, "call_type": "session_extraction"},
                    )
                    assignments = []
                    canonicalizations = []
                    candidate_assignments = []
                    for index, candidate in enumerate(output.new_target_candidates):
                        canonical_prompt = build_canonicalization_prompt(
                            prompts["canonicalize"],
                            registry,
                            candidate.model_dump(),
                            note.note_text,
                        )
                        decision, _ = call_with_one_repair(
                            model_client=model_client,
                            store=store,
                            system_prompt=prompts["system"],
                            task_prompt=canonical_prompt,
                            schema=CanonicalizationOutput,
                            schema_name="CanonicalizationOutput",
                            validate_context=lambda value, label=candidate.proposed_label: (
                                validate_canonicalization(value, label, registry)
                            ),
                            audit_context={
                                **audit_context,
                                "call_type": "canonicalization",
                                "candidate_index": index,
                            },
                        )
                        target_id = registry.apply_canonicalization(
                            candidate, decision, note.session_index
                        )
                        canonicalizations.append(decision.model_dump())
                        candidate_assignments.append(
                            {
                                "source_index": index,
                                "proposed_label": candidate.proposed_label,
                                "target_id": target_id,
                                "decision": decision.decision,
                            }
                        )
                    prior_record = _later_prior_record(output, candidate_assignments)

                envelope = {
                    **audit_context,
                    "completed_at": utc_now(),
                    "request_fingerprint": fingerprint,
                    "input_note_sha256": sha256_text(note.note_text),
                    "prompt_bundle_sha256": prompts_hash,
                    "input_tokens": input_tokens,
                    "model_output": output.model_dump(),
                    "target_assignments": assignments,
                    "canonicalizations": canonicalizations,
                    "candidate_assignments": candidate_assignments,
                    "registry_before": registry_before.model_dump(),
                    "registry_after": registry.model_dump(),
                    "prior_structured_record": prior_record,
                }
                store.append_validated(envelope)
                store.save_registry(client_id, registry.model_dump())
                latest[(client_id, note.session_index)] = envelope
                prior_records.append(prior_record)
                _update_observed(prior_observed, prior_record)
                store.materialize_parquet()
                counters["completed"] += 1
            except ContextOverflowError as exc:
                store.append_failure(
                    {
                        **audit_context,
                        "created_at": utc_now(),
                        "reason": "context_overflow",
                        "error": str(exc),
                    }
                )
                counters["context_overflow"] += 1
                break
            except Exception as exc:
                store.append_failure(
                    {
                        **audit_context,
                        "created_at": utc_now(),
                        "reason": "session_failed",
                        "error": str(exc),
                    }
                )
                counters["failed"] += 1
                break

    store.materialize_parquet()
    return counters


def _session1_prior_record(
    output: Session1Output, assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    records = []
    for assignment in assignments:
        target = output.targets[assignment["source_index"]]
        records.append(
            {
                "target_id": assignment["target_id"],
                "substantively_treated": target.substantively_treated,
                "performance_observed": target.performance_observed,
                "treatment_evidence": target.treatment_evidence,
                "performance_evidence": target.performance_evidence,
                "source": "initial_discovery",
            }
        )
    return {
        "client_id": output.client_id,
        "session_index": output.session_index,
        "note_id": output.note_id,
        "target_records": records,
    }


def _later_prior_record(
    output: LaterSessionOutput, candidate_assignments: list[dict[str, Any]]
) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {
        item.target_id: {
            "target_id": item.target_id,
            "substantively_treated": item.substantively_treated,
            "performance_observed": item.performance_observed,
            "treatment_evidence": list(item.treatment_evidence),
            "performance_evidence": list(item.performance_evidence),
            "source": "registered_assessment",
        }
        for item in output.target_assessments
    }
    for assignment in candidate_assignments:
        candidate = output.new_target_candidates[assignment["source_index"]]
        target_id = assignment["target_id"]
        if target_id not in by_target:
            by_target[target_id] = {
                "target_id": target_id,
                "substantively_treated": candidate.substantively_treated,
                "performance_observed": candidate.performance_observed,
                "treatment_evidence": (
                    candidate.evidence if candidate.substantively_treated else []
                ),
                "performance_evidence": (
                    candidate.evidence if candidate.performance_observed else []
                ),
                "source": "new_target_discovery",
            }
        elif assignment["decision"] == "same_as_existing":
            record = by_target[target_id]
            record["substantively_treated"] = max(
                record["substantively_treated"], candidate.substantively_treated
            )
            record["performance_observed"] = max(
                record["performance_observed"], candidate.performance_observed
            )
            if candidate.substantively_treated:
                record["treatment_evidence"] = _unique_strings(
                    [*record["treatment_evidence"], *candidate.evidence]
                )
            if candidate.performance_observed:
                record["performance_evidence"] = _unique_strings(
                    [*record["performance_evidence"], *candidate.evidence]
                )
    return {
        "client_id": output.client_id,
        "session_index": output.session_index,
        "note_id": output.note_id,
        "target_records": list(by_target.values()),
    }


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _update_observed(
    observed: dict[tuple[int, str], int], prior_record: dict[str, Any]
) -> None:
    session_index = int(prior_record["session_index"])
    for target in prior_record["target_records"]:
        observed[(session_index, target["target_id"])] = int(target["performance_observed"])


def selected_client_ids(args: argparse.Namespace) -> list[str]:
    values = list(args.client_id or [])
    if args.client_file:
        values.extend(
            line.strip()
            for line in args.client_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError("At least one --client-id or --client-file entry is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-data", help="Inspect and validate input data")
    inspect_parser.add_argument("--data-config", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="Run a selected-client extraction dry run")
    run_parser.add_argument("--data-config", type=Path, required=True)
    run_parser.add_argument("--model-config", type=Path, required=True)
    run_parser.add_argument("--client-id", action="append")
    run_parser.add_argument("--client-file", type=Path)
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=PACKAGE_ROOT / "outputs",
        help="Root under which the model-specific output directory is created",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data_config = load_data_config(args.data_config)
        if args.command == "inspect-data":
            print(json.dumps(inspect_input(data_config), indent=2, ensure_ascii=False))
            return 0

        clients = selected_client_ids(args)
        notes = load_notes(data_config, require_deidentified=True)
        model_config = load_model_config(args.model_config)
        result = process_clients(
            notes=notes,
            selected_clients=clients,
            model_config=model_config,
            output_root=args.output_root,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["failed"] or result["context_overflow"] else 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
