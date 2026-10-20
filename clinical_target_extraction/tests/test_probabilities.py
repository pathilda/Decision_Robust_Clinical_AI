"""Probability contracts, ordered model responses, exports and safe checkpoint reuse."""

import json
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError

from clinical_target_extraction.src.cli import build_parser
from clinical_target_extraction.src.model_client import CompletionResult, VLLMClient, repair_prompt
from clinical_target_extraction.src.prompts import build_classification_prompt, load_prompts
from clinical_target_extraction.src.run_classification import process_clients
from clinical_target_extraction.src.schemas import (
    PROBABILITY_COLUMNS,
    PROBABILITY_CONTRACT,
    PROBABILITY_STYLES,
    TREATMENT_CATEGORIES,
    ClassificationWithProbabilitiesOutput,
    parse_classification_output,
    schema_for_prompt_style,
)
from opendecider_inference.store import PROBABILITY_COLUMNS as OD_COLUMNS

from .test_classification import client, model_config


def response(client_id="001", reasoning=False):
    result = {
        "confidence": 0.65,
        "probabilities": dict(zip(TREATMENT_CATEGORIES, [0.15, 0.65, 0.20])),
        "client_id": client_id,
    }
    if reasoning:
        result["brief_reasoning"] = "Multiple documented needs require coordination."
    result["treatment_category"] = TREATMENT_CATEGORIES[1]
    return result


class ProbabilityClient:
    constructions = 0
    calls: ClassVar[list] = []

    def __init__(self, config):
        type(self).constructions += 1

    def complete_batch(self, requests):
        type(self).calls.append(requests)
        result = []
        for _, task, schema, _ in requests:
            cid = re.search(r"CLIENT_ID: ([^\n]+)", task).group(1)
            value = response(cid, "brief_reasoning" in schema.model_fields)
            result.append(CompletionResult(json.dumps(value), 30))
        return result

    def close(self):
        pass


def run(tmp_path, style="probabilities", clients=None):
    clients = clients or [client("001", 2)]
    return process_clients(
        clients=clients,
        selected_clients=[c.client_id for c in clients],
        model_config=model_config(),
        output_root=tmp_path,
        batch_size=2,
        prompt_style=style,
    )


@pytest.mark.parametrize("style", sorted(PROBABILITY_STYLES))
def test_schema_prompt_cli_order_and_unchanged_rubric(style):
    args = build_parser().parse_args(
        [
            "--pipeline",
            "classify",
            "--input",
            "clients.xlsx",
            "--prompt-style",
            style,
        ]
    )
    assert args.prompt_style == style
    schema = schema_for_prompt_style(style)
    payload = response(reasoning=style.startswith("brief"))
    output = parse_classification_output(json.dumps(payload), schema)
    assert list(schema.model_json_schema()["properties"]) == list(payload)
    assert list(output.model_dump(by_alias=True)) == list(payload)
    assert next(iter(payload)) == "confidence" and list(payload)[-1] == "treatment_category"
    base = load_prompts()["classification"]
    prompt = build_classification_prompt(load_prompts(style)["classification"], client("001", 2))
    for line in base.splitlines():
        if line.startswith('- "'):
            assert line in prompt
    assert "CLIENT_ID: 001" in prompt and "sum" in prompt
    assert PROBABILITY_COLUMNS == OD_COLUMNS


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, float("inf"), float("nan"), "0.65", True, None])
@pytest.mark.parametrize("field", ["confidence", "probability"])
def test_probability_numbers_are_strict_and_finite(field, bad_value):
    payload = response()
    if field == "confidence":
        payload[field] = bad_value
    else:
        payload["probabilities"][TREATMENT_CATEGORIES[1]] = bad_value
    with pytest.raises(ValidationError):
        ClassificationWithProbabilitiesOutput.model_validate(payload)


@pytest.mark.parametrize(
    "issue", ["sum", "confidence", "argmax", "missing", "extra", "label", "reason"]
)
def test_probability_consistency_and_exact_keys(issue):
    payload = response()
    if issue == "sum":
        payload["probabilities"][TREATMENT_CATEGORIES[0]] = 0.1
    elif issue == "confidence":
        payload["confidence"] = 0.8
    elif issue == "argmax":
        payload["treatment_category"] = TREATMENT_CATEGORIES[0]
    elif issue == "missing":
        del payload["probabilities"][TREATMENT_CATEGORIES[0]]
    elif issue == "extra":
        payload["probabilities"]["unknown"] = 0.0
    elif issue == "label":
        payload["treatment_category"] = "moderate"
    else:
        payload["brief_reasoning"] = "Unexpected in this mode"
    with pytest.raises(ValidationError):
        ClassificationWithProbabilitiesOutput.model_validate(payload)


def test_rounding_tolerance_no_normalization_and_deterministic_ties():
    payload = response()
    payload["probabilities"] = dict(zip(TREATMENT_CATEGORIES, [0.333333, 0.333333, 0.333333]))
    payload.update(confidence=0.333333, treatment_category=TREATMENT_CATEGORIES[0])
    output = ClassificationWithProbabilitiesOutput.model_validate(payload)
    assert output.probabilities.easy == 0.333333
    payload["treatment_category"] = TREATMENT_CATEGORIES[1]
    with pytest.raises(ValidationError, match="exact ties"):
        ClassificationWithProbabilitiesOutput.model_validate(payload)
    payload["treatment_category"] = TREATMENT_CATEGORIES[0]
    payload["probabilities"] = dict(zip(TREATMENT_CATEGORIES, [0.3333, 0.3333, 0.3333]))
    with pytest.raises(ValidationError, match="sum"):
        ClassificationWithProbabilitiesOutput.model_validate(payload)


def test_raw_order_duplicates_and_empty_reasoning_rejected():
    schema = ClassificationWithProbabilitiesOutput
    with pytest.raises(ValueError, match="exact order"):
        parse_classification_output(json.dumps(dict(reversed(list(response().items())))), schema)
    duplicate = json.dumps(response()).replace(
        '"confidence": 0.65', '"confidence": 0.65, "confidence": 0.65'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        parse_classification_output(duplicate, schema)
    nested_duplicate = json.dumps(response()).replace(
        '"Easy to treatment": 0.15', '"Easy to treatment": 0.15, "Easy to treatment": 0.15'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        parse_classification_output(nested_duplicate, schema)
    payload = response(reasoning=True)
    payload["brief_reasoning"] = "   "
    with pytest.raises(ValidationError):
        schema_for_prompt_style("brief-reasoning-probabilities").model_validate(payload)


@pytest.mark.parametrize("style", sorted(PROBABILITY_STYLES))
def test_pipeline_exports_raw_json_csv_excel_parquet_and_reuses(tmp_path, monkeypatch, style):
    ProbabilityClient.constructions, ProbabilityClient.calls = 0, []
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", ProbabilityClient
    )
    # Existing non-probability artifacts are not read or overwritten by the new styles.
    for name in ("qwen", "qwen_brief_reasoning"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "client_classifications.csv").write_text("existing result\n")
    result = run(tmp_path, style, [client("001", 2), client("002", 3)])
    assert result["completed"] == 2
    folder = tmp_path / ("qwen_" + style.replace("-", "_"))
    frames = [
        pd.read_csv(
            folder / "client_classifications.csv", dtype={"Client AlayaCare Client ID": str}
        ),
        pd.read_excel(
            folder / "client_classifications.xlsx", dtype={"Client AlayaCare Client ID": str}
        ),
    ]
    if (folder / "client_classifications.parquet").exists():
        frames.append(pd.read_parquet(folder / "client_classifications.parquet"))
    for frame in frames:
        assert frame["Client AlayaCare Client ID"].tolist() == ["001", "002"]
        assert frame.confidence.tolist() == [0.65, 0.65]
        assert frame.probability_easy.tolist() == [0.15, 0.15]
        assert frame.probability_moderate.tolist() == [0.65, 0.65]
        assert frame.probability_severe.tolist() == [0.20, 0.20]
        assert frame.probability_source.eq("llm_self_reported").all()
        assert not frame.probabilities_calibrated.any()
        assert frame.columns.get_loc("confidence") < frame.columns.get_loc("treatment_category")
    records = [
        json.loads(line)
        for line in (folder / "validated_classifications.jsonl").read_text().splitlines()
    ]
    for record in records:
        assert next(iter(record["model_output"])) == "confidence"
        assert list(record["model_output"])[-1] == "treatment_category"
        assert record["probabilities"] == record["model_output"]["probabilities"]
    raw = json.loads((folder / "raw_responses.jsonl").read_text().splitlines()[0])
    assert next(iter(json.loads(raw["raw_response"]))) == "confidence"
    config = json.loads((folder / "run_config.json").read_text())
    assert config["probability_contract"] == PROBABILITY_CONTRACT
    result = run(tmp_path, style, [client("001", 2), client("002", 3)])
    assert result["reused"] == 2 and ProbabilityClient.constructions == 1
    for name in ("qwen", "qwen_brief_reasoning"):
        assert (tmp_path / name / "client_classifications.csv").read_text() == "existing result\n"


@pytest.mark.parametrize("kind", ["sum", "order", "id"])
def test_bad_output_gets_one_repair_with_probability_instructions(tmp_path, monkeypatch, kind):
    class RepairClient(ProbabilityClient):
        count = 0

        def complete_batch(self, requests):
            self.count += 1
            payload = response()
            if self.count == 1:
                if kind == "sum":
                    payload["probabilities"][TREATMENT_CATEGORIES[0]] = 0.1
                elif kind == "id":
                    payload["client_id"] = "different"
                else:
                    payload = dict(reversed(list(payload.items())))
            else:
                assert "REPAIR INVALID STRUCTURED OUTPUT" in requests[0][1]
                assert "Do not retain an inconsistent category" in requests[0][1]
            return [CompletionResult(json.dumps(payload), 40)]

    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", RepairClient
    )
    assert run(tmp_path)["completed"] == 1
    raw = [
        json.loads(line)
        for line in (tmp_path / "qwen_probabilities/raw_responses.jsonl").read_text().splitlines()
    ]
    assert [(r["attempt"], r["valid"]) for r in raw] == [("initial", False), ("repair", True)]
    assert all("request_fingerprint" in r for r in raw)


def test_failed_changed_request_never_exports_old_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", ProbabilityClient
    )
    run(tmp_path)

    class BadClient(ProbabilityClient):
        calls = 0

        def complete_batch(self, requests):
            type(self).calls += 1
            return [CompletionResult(json.dumps(response() | {"confidence": 0.9}), 20)]

    monkeypatch.setattr("clinical_target_extraction.src.run_classification.VLLMClient", BadClient)
    changed = client("001", 2)
    changed = type(changed)(**{**changed.__dict__, "sp_text": "changed input"})
    result = run(tmp_path, clients=[changed])
    assert result["failed"] == 1 and BadClient.calls == 2
    folder = tmp_path / "qwen_probabilities"
    assert pd.read_csv(folder / "client_classifications.csv").empty
    assert len((folder / "validated_classifications.jsonl").read_text().splitlines()) == 1
    failure = json.loads((folder / "failures.jsonl").read_text())
    assert failure["request_fingerprint"]


def test_probability_modes_have_distinct_fingerprints_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", ProbabilityClient
    )
    fingerprints = []
    for style in sorted(PROBABILITY_STYLES):
        assert run(tmp_path, style)["completed"] == 1
        folder = tmp_path / ("qwen_" + style.replace("-", "_"))
        record = json.loads((folder / "validated_classifications.jsonl").read_text())
        fingerprints.append(record["request_fingerprint"])
    assert len(set(fingerprints)) == 2


def test_repair_for_standard_prompt_stays_classification_preserving():
    from clinical_target_extraction.src.schemas import ClassificationOutput

    text = repair_prompt("task", "bad", ["error"], ClassificationOutput)
    assert "preserving the model's substantive classification" in text
    assert "Reassess the distribution" not in text


def test_structured_backend_receives_confidence_first_schema(monkeypatch):
    """Exercise the real batching adapter without importing a GPU engine."""
    vllm = ModuleType("vllm")
    sampling = ModuleType("vllm.sampling_params")
    vllm.SamplingParams = SimpleNamespace
    sampling.StructuredOutputsParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2, 3] if tokenize else str(messages)

    class Engine:
        def generate(self, prompts, sampling_params, use_tqdm):
            outputs = []
            for params in sampling_params:
                schema = params.structured_outputs.json
                assert next(iter(schema["properties"])) == "confidence"
                assert list(schema["properties"])[-1] == "treatment_category"
                assert schema["additionalProperties"] is False
                assert set(schema["$defs"]["CategoryProbabilities"]["properties"]) == set(
                    TREATMENT_CATEGORIES
                )
                text = json.dumps(response(reasoning="brief_reasoning" in schema["properties"]))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
            return outputs

    backend = VLLMClient(model_config(), llm=Engine(), tokenizer=Tokenizer())
    requests = [
        ("system", "task", schema_for_prompt_style(s), s) for s in sorted(PROBABILITY_STYLES)
    ]
    completions = backend.complete_batch(requests)
    assert len(completions) == 2
    for completion, request in zip(completions, requests):
        assert completion.input_tokens == 3 and completion.error is None
        parse_classification_output(completion.text, request[2])


def test_invalid_matching_probability_checkpoint_is_recomputed(tmp_path, monkeypatch):
    ProbabilityClient.constructions = 0
    monkeypatch.setattr(
        "clinical_target_extraction.src.run_classification.VLLMClient", ProbabilityClient
    )
    run(tmp_path)
    path = tmp_path / "qwen_probabilities/validated_classifications.jsonl"
    record = json.loads(path.read_text())
    record["model_output"]["confidence"] = 0.99
    path.write_text(json.dumps(record) + "\n")
    result = run(tmp_path)
    assert result["reused"] == 0 and result["completed"] == 1
    assert ProbabilityClient.constructions == 2
    assert pd.read_csv(path.parent / "client_classifications.csv").confidence.iloc[0] == 0.65
