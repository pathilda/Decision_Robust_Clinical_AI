from __future__ import annotations

from pathlib import Path

import pytest

from clinical_target_extraction.src.cli import _model_names, build_parser
from clinical_target_extraction.src.settings import resolve_model_path


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


def test_default_batch_size_enables_direct_vllm_batching() -> None:
    args = build_parser().parse_args(["--pipeline", "extract"])
    assert args.batch_size == 8


def test_concurrency_is_kept_as_a_compatibility_alias() -> None:
    args = build_parser().parse_args(
        ["--pipeline", "extract", "--concurrency", "4"]
    )
    assert args.batch_size == 4
