"""Load and render versioned plain-text prompt templates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .data import ClinicalNote
from .registry import TargetRegistry


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = {
    "system": "system.txt",
    "session1": "session1.txt",
    "later_session": "later_session.txt",
    "canonicalize": "canonicalize.txt",
}
PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def load_prompts(prompt_dir: Path = PROMPT_DIR) -> dict[str, str]:
    return {
        name: (prompt_dir / filename).read_text(encoding="utf-8").strip()
        for name, filename in PROMPT_FILES.items()
    }


def prompt_bundle_hash(prompts: dict[str, str]) -> str:
    payload = json.dumps(prompts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render(template: str, **values: Any) -> str:
    expected = set(PLACEHOLDER.findall(template))
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Prompt values mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    def replacement(match: re.Match[str]) -> str:
        return str(values[match.group(1)])

    rendered = PLACEHOLDER.sub(replacement, template)
    if PLACEHOLDER.search(rendered):
        raise ValueError("Unrendered prompt placeholders remain")
    return rendered


def build_session1_prompt(template: str, note: ClinicalNote) -> str:
    return render(
        template,
        client_id=note.client_id,
        note_id=note.note_id,
        note_text=note.note_text,
    )


def note_boundaries(notes: list[ClinicalNote]) -> str:
    return "\n".join(
        f'<SESSION index="{note.session_index}" note_id="{note.note_id}">\n'
        f"{note.note_text}\n</SESSION>"
        for note in notes
    )


def build_later_prompt(
    template: str,
    registry: TargetRegistry,
    notes_to_date: list[ClinicalNote],
    prior_records: list[dict[str, Any]],
) -> str:
    current = notes_to_date[-1]
    return render(
        template,
        client_id=current.client_id,
        current_session_index=current.session_index,
        current_note_id=current.note_id,
        target_registry_json=json.dumps(
            registry.model_dump(), indent=2, ensure_ascii=False
        ),
        prior_outputs_json=json.dumps(prior_records, indent=2, ensure_ascii=False),
        notes_with_explicit_session_boundaries=note_boundaries(notes_to_date),
    )


def build_canonicalization_prompt(
    template: str,
    registry: TargetRegistry,
    candidate: dict[str, Any],
    current_note_text: str,
) -> str:
    return render(
        template,
        target_registry_json=json.dumps(registry.model_dump(), indent=2, ensure_ascii=False),
        candidate_json=json.dumps(candidate, indent=2, ensure_ascii=False),
        current_note_text=current_note_text,
    )
