from __future__ import annotations

from pathlib import Path

import pytest

from clinical_target_extraction.src.cli import _model_names, build_parser
from clinical_target_extraction.src.server import build_server_command
from clinical_target_extraction.src.settings import MODEL_PRESETS, resolve_model_path


def test_all_models_have_expected_execution_order() -> None:
    args = build_parser().parse_args(["--pipeline", "extract", "--model", "all"])
    assert _model_names(args) == ["qwen", "medgemma", "gpt_oss"]


def test_model_path_uses_expected_download_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Qwen"
    checkpoint.mkdir()
    assert resolve_model_path("qwen", tmp_path) == checkpoint.resolve()


def test_missing_model_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model directory does not exist"):
        resolve_model_path("medgemma", tmp_path)


def test_vllm_command_uses_local_path_and_model_preset(tmp_path: Path) -> None:
    model_path = tmp_path / "GPT_OSS"
    command = build_server_command(
        preset=MODEL_PRESETS["gpt_oss"],
        model_path=model_path,
        host="127.0.0.1",
        port=8000,
        max_model_len=32768,
        gpu_memory_utilization=0.92,
    )
    assert command[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--model") + 1] == str(model_path)
    assert command[command.index("--dtype") + 1] == "auto"
    assert command[command.index("--served-model-name") + 1] == "extractor"
