"""Direct batched offline inference through vLLM's Python API."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizerBase


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_label: str
    model_id: str
    tokenizer_id: str
    dtype: str = "auto"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = Field(default=32768, gt=0)
    seed: int = 20260924
    max_model_len: int = Field(default=131072, gt=0)
    gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)
    max_num_seqs: int = Field(default=8, gt=0)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ContextOverflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionResult:
    text: str | None
    input_tokens: int
    error: Exception | None = None


class VLLMClient:
    """Load one model in-process and submit prompt lists to ``LLM.generate``."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        llm: Any | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config
        if llm is None:
            # Set before importing vLLM so Alliance nodes do not try to JIT
            # compile the FlashInfer sampler without nvcc.
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
            from vllm import LLM

            print(f"\n>>> Loading {config.model_label} directly from {config.model_id}")
            print(">>> Offline vLLM model-loading progress follows:\n", flush=True)
            llm = LLM(
                model=config.model_id,
                tokenizer=config.tokenizer_id,
                dtype=config.dtype,
                tensor_parallel_size=1,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_num_seqs=config.max_num_seqs,
                seed=config.seed,
                generation_config="vllm",
            )
        self.llm = llm
        self.tokenizer = tokenizer or llm.get_tokenizer()

    def _render(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_tokens = len(token_ids)
        required = input_tokens + self.config.max_tokens
        if required > self.config.max_model_len:
            raise ContextOverflowError(
                f"Request needs {required} tokens ({input_tokens} input + "
                f"{self.config.max_tokens} reserved output), exceeding max_model_len="
                f"{self.config.max_model_len}"
            )
        return str(rendered), input_tokens

    def complete_batch(
        self,
        requests: list[tuple[str, str, type[BaseModel], str]],
    ) -> list[CompletionResult]:
        """Generate a request list in one direct, progress-visible vLLM batch."""

        if not requests:
            return []
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        results: list[CompletionResult | None] = [None] * len(requests)
        prompts: list[str] = []
        sampling_params: list[Any] = []
        valid_indices: list[int] = []
        input_counts: dict[int, int] = {}
        for index, (system_prompt, user_prompt, schema, _schema_name) in enumerate(requests):
            try:
                rendered, input_tokens = self._render(system_prompt, user_prompt)
            except ContextOverflowError as exc:
                results[index] = CompletionResult(None, 0, exc)
                continue
            prompts.append(rendered)
            input_counts[index] = input_tokens
            valid_indices.append(index)
            sampling_params.append(
                SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    seed=self.config.seed,
                    max_tokens=self.config.max_tokens,
                    structured_outputs=StructuredOutputsParams(
                        json=schema.model_json_schema()
                    ),
                )
            )

        if prompts:
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=True,
            )
            if len(outputs) != len(valid_indices):
                raise RuntimeError(
                    "vLLM returned a different number of outputs than submitted prompts"
                )
            for index, output in zip(valid_indices, outputs, strict=True):
                if not output.outputs:
                    results[index] = CompletionResult(
                        None,
                        input_counts[index],
                        RuntimeError("Model returned no completion candidates"),
                    )
                else:
                    results[index] = CompletionResult(
                        output.outputs[0].text,
                        input_counts[index],
                    )
        return [
            result
            if result is not None
            else CompletionResult(None, 0, RuntimeError("Missing generation result"))
            for result in results
        ]

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        schema_name: str,
    ) -> tuple[str, int]:
        """Compatibility wrapper for a single direct generation request."""

        result = self.complete_batch([(system_prompt, user_prompt, schema, schema_name)])[0]
        if result.error is not None:
            raise result.error
        if result.text is None:
            raise RuntimeError("Model returned an empty message")
        return result.text, result.input_tokens

    def close(self) -> None:
        """Release the offline engine before another model is loaded."""

        engine = getattr(self.llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.llm = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def repair_prompt(
    original_task: str,
    invalid_response: str,
    validation_errors: list[str],
    schema: type[BaseModel],
) -> str:
    """Build the sole allowed repair request without modifying any output in Python."""

    return (
        "TASK: REPAIR INVALID STRUCTURED OUTPUT\n\n"
        "Return a corrected response for the original task. Correct structural and schema "
        "errors while preserving the model's substantive binary judgments. Return only compact "
        "JSON matching the schema.\n\n"
        f"<VALIDATION_ERRORS>\n{json.dumps(validation_errors, indent=2)}\n"
        "</VALIDATION_ERRORS>\n\n"
        f"<INVALID_RESPONSE>\n{invalid_response}\n</INVALID_RESPONSE>\n\n"
        f"<REQUIRED_SCHEMA>\n{json.dumps(schema.model_json_schema(), indent=2)}\n"
        "</REQUIRED_SCHEMA>\n\n"
        f"<ORIGINAL_TASK>\n{original_task}\n</ORIGINAL_TASK>"
    )
