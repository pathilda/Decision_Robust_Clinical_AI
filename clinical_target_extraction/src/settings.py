"""Central model presets used by the command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPreset:
    name: str
    directory: str
    dtype: str


MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen": ModelPreset("qwen", "Qwen", "bfloat16"),
    "medgemma": ModelPreset("medgemma", "MedGemma", "bfloat16"),
    "gpt_oss": ModelPreset("gpt_oss", "GPT_OSS", "auto"),
}


def resolve_model_path(
    model_name: str,
    model_root: Path,
    model_path: Path | None = None,
) -> Path:
    """Return and validate the local checkpoint directory for one preset."""

    if model_path is not None:
        path = model_path.expanduser().resolve()
    else:
        path = (model_root.expanduser() / MODEL_PRESETS[model_name].directory).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    return path
