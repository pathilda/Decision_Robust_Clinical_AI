"""Strictly local OpenDecider loading with an explicit base-model override."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


class LocalOpenDecider:
    """Load a local adapter against a separately downloaded local base model."""

    def __init__(self, adapter_path: Path, base_path: Path, *, device: str = "cuda") -> None:
        self.adapter_path = adapter_path.expanduser().resolve()
        self.base_path = base_path.expanduser().resolve()
        _validate_model_paths(self.adapter_path, self.base_path)
        for name, value in OFFLINE_ENVIRONMENT.items():
            os.environ[name] = value

        self._overlay = tempfile.TemporaryDirectory(prefix="opendecider-local-")
        overlay_path = Path(self._overlay.name)
        _build_adapter_overlay(
            source=self.adapter_path,
            destination=overlay_path,
            base_path=self.base_path,
        )
        try:
            from opendecider import load
        except ImportError as exc:
            self._overlay.cleanup()
            raise RuntimeError(
                "OpenDecider is not installed; run "
                "pip install -e \".[opendecider]\""
            ) from exc
        try:
            self.model = load(str(overlay_path), device=device)
        except Exception:
            self._overlay.cleanup()
            raise

    def classify_batch(
        self,
        states: list[str],
        questions: dict[str, dict[str, object]],
    ) -> list[dict[str, Any]]:
        batch_method = getattr(self.model, "system_one_batch", None)
        if callable(batch_method):
            return list(batch_method(states, questions))
        return [self.model.system_one(state, questions) for state in states]

    def close(self) -> None:
        self.model = None
        self._overlay.cleanup()
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _validate_model_paths(adapter_path: Path, base_path: Path) -> None:
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"OpenDecider adapter directory does not exist: {adapter_path}")
    if not base_path.is_dir():
        raise FileNotFoundError(f"OpenDecider base directory does not exist: {base_path}")
    required = {
        adapter_path / "opendecider.json": "adapter metadata",
        adapter_path / "adapter_config.json": "PEFT adapter configuration",
        base_path / "config.json": "base-model configuration",
    }
    missing = [
        f"{description}: {path}"
        for path, description in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing local model files: " + "; ".join(missing))

    adapter_weights = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    )
    if not any(path.is_file() for path in adapter_weights):
        raise FileNotFoundError(
            "The adapter folder must contain adapter_model.safetensors or adapter_model.bin: "
            f"{adapter_path}"
        )


def _build_adapter_overlay(*, source: Path, destination: Path, base_path: Path) -> None:
    """Create a temporary symlink overlay with local base metadata."""

    for child in source.iterdir():
        if child.name == "opendecider.json":
            continue
        target = destination / child.name
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())

    metadata_path = source / "opendecider.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid OpenDecider metadata in {metadata_path}: {exc}") from exc
    if metadata.get("kind") != "small":
        raise ValueError(
            "Expected an OpenDecider PEFT decision adapter with kind='small'; "
            f"got {metadata.get('kind')!r}"
        )
    metadata["base_model"] = str(base_path)
    (destination / "opendecider.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
