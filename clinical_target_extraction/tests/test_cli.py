from __future__ import annotations

from pathlib import Path

import pytest

from clinical_target_extraction.src.cli import _model_names, build_parser
from clinical_target_extraction.src.settings import resolve_model_path


def parse(*values: str):
    return build_parser().parse_args(
        ["--pipeline", "classify", "--input", "clients.xlsx", *values]
    )


def test_cli_uses_new_workbook_column_defaults() -> None:
    args = parse()
    assert args.id_column == "Client AlayaCare Client ID"
    assert args.sp_column == "SP text"
    assert args.assessment_column == "assessment text"
    assert args.max_tokens == 256
    assert args.prompt_style == "standard"


def test_brief_reasoning_prompt_is_selectable() -> None:
    assert parse("--prompt-style", "brief-reasoning").prompt_style == "brief-reasoning"


def test_all_models_have_expected_execution_order() -> None:
    assert _model_names(parse("--model", "all")) == ["qwen", "medgemma", "gpt_oss"]


def test_model_path_uses_expected_download_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Qwen"
    checkpoint.mkdir()
    assert resolve_model_path("qwen", tmp_path) == checkpoint.resolve()


def test_missing_model_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model directory does not exist"):
        resolve_model_path("medgemma", tmp_path)


def test_default_batch_size_is_eight() -> None:
    assert parse().batch_size == 8


def test_concurrency_is_a_compatibility_alias() -> None:
    assert parse("--concurrency", "4").batch_size == 4
