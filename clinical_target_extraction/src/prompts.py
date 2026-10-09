"""Load and render the versioned client-classification prompts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .data import ClientNote


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_STYLES = {
    "standard": {
        "system": "classification_system.txt",
        "classification": "classification.txt",
    },
    "brief-reasoning": {
        "system": "classification_brief_reasoning_system.txt",
        "classification": "classification_brief_reasoning.txt",
    },
}
PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def load_prompts(
    prompt_style: str = "standard",
    prompt_dir: Path = PROMPT_DIR,
) -> dict[str, str]:
    try:
        prompt_files = PROMPT_STYLES[prompt_style]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt style {prompt_style!r}; choose from {sorted(PROMPT_STYLES)}"
        ) from exc
    return {
        name: (prompt_dir / filename).read_text(encoding="utf-8").strip()
        for name, filename in prompt_files.items()
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


def build_classification_prompt(template: str, client: ClientNote) -> str:
    return render(
        template,
        client_id=client.client_id,
        combined_note=client.note_text,
    )
