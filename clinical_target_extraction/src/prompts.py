"""Load and render versioned prompt templates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .data import ClinicalNote


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = {
    "system": "system.txt",
    "session1": "session1.txt",
    "later_session": "later_session.txt",
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
    return PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)


def build_session1_prompt(template: str, note: ClinicalNote) -> str:
    return render(
        template,
        client_id=note.client_id,
        note_id=note.note_id,
        note_text=note.note_text,
    )


def build_later_prompt(
    template: str,
    previous_profile: dict[str, Any],
    current_note: ClinicalNote,
) -> str:
    """Provide exactly one prior JSON snapshot plus the current raw note."""

    return render(
        template,
        client_id=current_note.client_id,
        current_session_index=current_note.session_index,
        current_note_id=current_note.note_id,
        previous_session_profile_json=json.dumps(
            previous_profile,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        current_note_text=current_note.note_text,
    )
