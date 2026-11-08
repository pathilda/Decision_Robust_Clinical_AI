"""outputs/manifest.json (spec Section 9): versions, revisions, settings and hashes."""

from __future__ import annotations

import json
import platform

from .common import (
    CONFIG_DIR,
    SCHEMAS_DIR,
    file_sha256,
    models_config,
    now,
    prompt_hashes,
    prompt_version,
    study_config,
    write_json_atomic,
)


def update(ws, model=None, engine_info=None):
    """Merge the shared section and (optionally) one model's engine section."""
    manifest = {}
    if ws.manifest.is_file():
        manifest = json.loads(ws.manifest.read_text(encoding="utf-8"))
    cfg = models_config()
    manifest.update(
        updated_at=now(),
        python_version=platform.python_version(),
        serving_settings=cfg["serving"],
        sampling_settings=cfg["sampling"],
        max_new_tokens=cfg["max_new_tokens"],
        chunk_file_sha256=file_sha256(ws.chunks_file) if ws.chunks_file.is_file() else None,
        class_config_sha256=file_sha256(CONFIG_DIR / "classes.yaml"),
        models_config_sha256=file_sha256(CONFIG_DIR / "models.yaml"),
        study_config_sha256=file_sha256(CONFIG_DIR / "study.yaml"),
        load_profiles_sha256=file_sha256(CONFIG_DIR / study_config()["load_profiles"]),
        prompt_template_version=prompt_version(),
        prompt_template_sha256=prompt_hashes(),
        schema_sha256={p.name: file_sha256(p) for p in sorted(SCHEMAS_DIR.glob("*.json"))},
    )
    if model and engine_info:
        models = manifest.setdefault("models", {})
        previous = models.get(model, {})
        if previous.get("revision") not in (None, engine_info.get("revision")):
            engine_info["revision_changed_from"] = previous["revision"]
        models[model] = {**engine_info, "recorded_at": now()}
        for key in ("vllm_version", "cuda_version", "gpu", "torch_version"):
            if engine_info.get(key) is not None:
                manifest[key] = engine_info[key]
        manifest["prefix_caching_active"] = {
            k: v.get("prefix_caching_active") for k, v in models.items()
        }
    write_json_atomic(ws.manifest, manifest)
    return manifest
