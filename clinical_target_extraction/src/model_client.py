"""Small adapter for a local vLLM OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from openai import OpenAI


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_label: str
    model_id: str
    served_model_name: str = "extractor"
    tokenizer_id: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "local-vllm"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = Field(default=6144, gt=0)
    seed: int = 20260924
    max_model_len: int = Field(default=32768, gt=0)
    structured_output_mode: Literal["response_format", "guided_json"] = "response_format"


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ContextOverflowError(RuntimeError):
    pass


class VLLMClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        openai_client: OpenAI | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config
        if openai_client is None:
            from openai import OpenAI

            openai_client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.client = openai_client
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.tokenizer_id)

    def token_count(self, system_prompt: str, user_prompt: str) -> int:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    def ensure_fits(self, system_prompt: str, user_prompt: str) -> int:
        input_tokens = self.token_count(system_prompt, user_prompt)
        required = input_tokens + self.config.max_tokens
        if required > self.config.max_model_len:
            raise ContextOverflowError(
                f"Request needs {required} tokens ({input_tokens} input + "
                f"{self.config.max_tokens} reserved output), exceeding max_model_len="
                f"{self.config.max_model_len}"
            )
        return input_tokens

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        schema_name: str,
    ) -> tuple[str, int]:
        input_tokens = self.ensure_fits(system_prompt, user_prompt)
        schema_json = schema.model_json_schema()
        kwargs: dict[str, Any] = {
            "model": self.config.served_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.structured_output_mode == "response_format":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema_json},
            }
        else:
            kwargs["extra_body"] = {"guided_json": schema_json}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Model returned an empty message")
        return content, input_tokens


def repair_prompt(
    original_task: str,
    invalid_response: str,
    validation_errors: list[str],
    schema: type[BaseModel],
) -> str:
    """Build the sole allowed repair request without modifying any output in Python."""

    return (
        "TASK: REPAIR INVALID STRUCTURED OUTPUT\n\n"
        "Return a corrected response for the original task. Correct every validation error. "
        "Return only JSON matching the schema. Evidence must remain an exact quotation from "
        "the notes in ORIGINAL_TASK; do not invent or paraphrase a quotation.\n\n"
        f"<VALIDATION_ERRORS>\n{json.dumps(validation_errors, indent=2)}\n"
        "</VALIDATION_ERRORS>\n\n"
        f"<INVALID_RESPONSE>\n{invalid_response}\n</INVALID_RESPONSE>\n\n"
        f"<REQUIRED_SCHEMA>\n{json.dumps(schema.model_json_schema(), indent=2)}\n"
        "</REQUIRED_SCHEMA>\n\n"
        f"<ORIGINAL_TASK>\n{original_task}\n</ORIGINAL_TASK>"
    )
