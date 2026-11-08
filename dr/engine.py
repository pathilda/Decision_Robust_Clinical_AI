"""Offline vLLM engine, batched structured generation, retries and records (spec 1.2, 8, 9)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import file_sha256, models_config, now, prompt_version
from .prompts import prompt_hash


class ParseError(ValueError):
    pass


@dataclass
class Request:
    """One inference. ``parse(obj) -> (parsed, flags, extra)`` raises ParseError if invalid."""

    key: str
    messages: list
    schema: dict
    max_tokens: int
    parse: Callable[[Any], tuple]
    meta: dict
    extra: dict = field(default_factory=dict)

    @property
    def prompt_hash(self):
        return prompt_hash(self.messages)


@dataclass
class Generation:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


# ----------------------------------------------------------------------------- engines
def _structured(schema):
    from vllm import SamplingParams  # noqa: F401  (import check)

    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}


def is_multimodal(model_path):
    config = Path(model_path) / "config.json"
    if not config.is_file():
        return False
    return "vision_config" in json.loads(config.read_text(encoding="utf-8"))


class VLLMEngine:
    """Loads one model once; every call uses llm.chat with the model's own chat template."""

    def __init__(self, entry, revision):
        cfg = models_config()
        self.entry, self.revision = entry, revision
        self.serving, self.sampling = dict(cfg["serving"]), dict(cfg["sampling"])
        self.chat_template_kwargs = dict(entry.get("chat_template_kwargs") or {})
        self.max_model_len = int(self.serving["max_model_len"])
        if entry["path"] is None:
            raise ValueError("LLM steps need --models-dir")
        path = Path(entry["path"])
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{entry['key']}: no checkpoint at {path}")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        self.multimodal = is_multimodal(path)
        kwargs = dict(
            model=str(path),
            tokenizer=str(path),
            dtype=self.serving["dtype"],
            max_model_len=self.max_model_len,
            gpu_memory_utilization=float(self.serving["gpu_memory_utilization"]),
            enable_prefix_caching=bool(self.serving["enable_prefix_caching"]),
            kv_cache_dtype=self.serving["kv_cache_dtype"],
            seed=int(self.serving["seed"]),
            tensor_parallel_size=int(self.serving.get("tensor_parallel_size", 1)),
            generation_config="vllm",
        )
        if self.multimodal:
            kwargs["limit_mm_per_prompt"] = dict(self.serving["limit_mm_per_prompt"])
        self.prefix_caching_error = None
        print(f">>> Loading {entry['key']} from {path}", file=sys.stderr, flush=True)
        try:
            self.llm = LLM(**kwargs)
        except Exception as exc:  # retry only for prefix-caching incompatibility
            if not kwargs["enable_prefix_caching"] or "prefix" not in str(exc).lower():
                raise
            self.prefix_caching_error = str(exc)[:500]
            print(f"warning: prefix caching unavailable ({exc}); retrying without", file=sys.stderr)
            kwargs["enable_prefix_caching"] = False
            self.llm = LLM(**kwargs)
        self.kwargs = kwargs
        self.tokenizer = self.llm.get_tokenizer()
        self.thinking_check = self._check_thinking()

    def _render(self, msgs, **overrides):
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            **{**self.chat_template_kwargs, **overrides},
        )

    def _check_thinking(self):
        if "enable_thinking" not in self.chat_template_kwargs:
            return "not applicable"
        probe = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        off = self._render(probe)
        on = self._render(probe, enable_thinking=True)
        if off == on:
            raise RuntimeError(
                f"{self.entry['key']}: chat template ignores enable_thinking; thinking cannot be "
                "disabled with the configured kwarg"
            )
        return "enable_thinking=False changes the rendered template"

    def count_tokens(self, msgs):
        return len(self.tokenizer.encode(self._render(msgs), add_special_tokens=False))

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def prefix_caching_active(self):
        engine = getattr(self.llm, "llm_engine", None)
        for path in ("vllm_config.cache_config", "cache_config"):
            obj = engine
            for part in path.split("."):
                obj = getattr(obj, part, None)
            if obj is not None and hasattr(obj, "enable_prefix_caching"):
                return bool(obj.enable_prefix_caching)
        return bool(self.kwargs["enable_prefix_caching"])

    def generate(self, items):
        """items: [(messages, schema, max_tokens, seed)] -> [Generation] in the same order."""
        from vllm import SamplingParams

        params = [
            SamplingParams(
                temperature=float(self.sampling["temperature"]),
                top_p=float(self.sampling["top_p"]),
                seed=int(seed),
                max_tokens=int(max_tokens),
                **_structured(schema),
            )
            for _, schema, max_tokens, seed in items
        ]
        outputs = self.llm.chat(
            [msgs for msgs, *_ in items],
            sampling_params=params,
            use_tqdm=True,
            chat_template_kwargs=self.chat_template_kwargs or None,
        )
        if len(outputs) != len(items):
            raise RuntimeError("vLLM returned a different number of outputs than requests")
        result = []
        for out in outputs:
            if not out.outputs:
                result.append(Generation("", "no_output", len(out.prompt_token_ids or []), 0))
                continue
            o = out.outputs[0]
            result.append(
                Generation(o.text, o.finish_reason, len(out.prompt_token_ids or []), len(o.token_ids))
            )
        return result

    def info(self):
        import vllm

        info = {
            "vllm_version": vllm.__version__,
            "model_path": str(self.entry["path"]),
            "hf_id": self.entry["hf_id"],
            "revision": self.revision,
            "multimodal_checkpoint": self.multimodal,
            "engine_kwargs": {k: v for k, v in self.kwargs.items() if k not in ("model", "tokenizer")},
            "prefix_caching_requested": bool(self.serving["enable_prefix_caching"]),
            "prefix_caching_active": self.prefix_caching_active(),
            "prefix_caching_error": self.prefix_caching_error,
            "chat_template_kwargs": self.chat_template_kwargs,
            "thinking_check": self.thinking_check,
        }
        config = Path(self.entry["path"]) / "config.json"
        info["config_sha256"] = file_sha256(config)
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            pass
        return info

    def close(self):
        engine = getattr(self.llm, "llm_engine", None)
        if callable(getattr(engine, "shutdown", None)):
            engine.shutdown()
        self.llm = None


class FakeEngine:
    """Deterministic stand-in for tests and dry runs: responder(messages, schema) -> text."""

    def __init__(self, responder, max_model_len=32768, chars_per_token=4):
        self.responder = responder
        self.max_model_len = max_model_len
        self.chars_per_token = chars_per_token
        self.calls = []

    def count_text(self, text):
        return max(1, len(text) // self.chars_per_token)

    def count_tokens(self, msgs):
        return sum(self.count_text(m["content"]) for m in msgs) + 10

    def generate(self, items):
        self.calls.append(items)
        result = []
        for msgs, schema, max_tokens, seed in items:
            text = self.responder(msgs, schema, max_tokens, seed)
            finish = "length" if self.count_text(text) > max_tokens else "stop"
            result.append(Generation(text, finish, self.count_tokens(msgs), self.count_text(text)))
        return result

    def info(self):
        return {"engine": "fake"}


# ----------------------------------------------------------------------------- runner
def _record(request, revision, **values):
    return {
        "key": request.key,
        **request.meta,
        "model_revision": revision,
        "prompt_template_version": prompt_version(),
        "prompt_hash": request.prompt_hash,
        "prompt_tokens": None,
        "completion_tokens": None,
        "raw_text": None,
        "parsed": None,
        "parse_ok": False,
        "attempts": 0,
        "finish_reason": None,
        "skipped": False,
        "skipped_reason": None,
        "flags": [],
        "errors": [],
        **request.extra,
        **values,
        "timestamp": now(),
    }


def _attempt_plan(base_tokens, seed, room):
    """Attempt 1: base. Attempt 2: max_tokens doubled. Attempt 3: also seed + 1."""
    doubled = min(2 * base_tokens, room)
    return [(1, min(base_tokens, room), seed), (2, doubled, seed), (3, doubled, seed + 1)]


def run_requests(engine, requests, store, revision, batch_size=512, seed=None):
    """Generate every request not already done in ``store`` (resume) and persist records.

    Returns a summary dict. Prompts are never truncated: a prompt that leaves fewer than
    max_new_tokens of context is recorded with skipped_reason="prompt_too_long".
    """
    seed = int(models_config()["sampling"]["seed"] if seed is None else seed)
    todo = [r for r in requests if not store.done(r.key, r.prompt_hash)]
    summary = {"requested": len(requests), "reused": len(requests) - len(todo), "ok": 0,
               "failed": 0, "prompt_too_long": 0}
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        live, records = [], []
        for request in batch:
            tokens = engine.count_tokens(request.messages)
            room = engine.max_model_len - tokens
            if room < request.max_tokens:
                records.append(
                    _record(request, revision, prompt_tokens=tokens, skipped=True,
                            skipped_reason="prompt_too_long",
                            errors=[f"prompt {tokens} + max_new_tokens {request.max_tokens} "
                                    f"> max_model_len {engine.max_model_len}"])
                )
                summary["prompt_too_long"] += 1
                continue
            live.append((request, _attempt_plan(request.max_tokens, seed, room), []))
        store.append(records)
        for attempt in range(3):
            if not live:
                break
            generations = engine.generate(
                [(r.messages, r.schema, plan[attempt][1], plan[attempt][2]) for r, plan, _ in live]
            )
            remaining, records = [], []
            for (request, plan, history), gen in zip(live, generations):
                number, max_tokens, used_seed = plan[attempt]
                entry = {
                    "attempt": number,
                    "max_tokens": max_tokens,
                    "seed": used_seed,
                    "finish_reason": gen.finish_reason,
                }
                try:
                    if gen.finish_reason == "length":
                        raise ParseError("truncated: finish_reason == 'length'")
                    try:
                        obj = json.loads(gen.text)
                    except json.JSONDecodeError as exc:
                        raise ParseError(f"invalid JSON: {exc}") from exc
                    parsed, flags, extra = request.parse(obj)
                except ParseError as exc:
                    entry["error"] = str(exc)
                    history.append(entry)
                    if number == 3:
                        records.append(
                            _record(request, revision, prompt_tokens=gen.prompt_tokens,
                                    completion_tokens=gen.completion_tokens, raw_text=gen.text,
                                    attempts=number, finish_reason=gen.finish_reason,
                                    errors=[h["error"] for h in history], attempt_log=history,
                                    max_tokens=max_tokens, seed_used=used_seed)
                        )
                        summary["failed"] += 1
                    else:
                        remaining.append((request, plan, history))
                    continue
                history.append(entry)
                records.append(
                    _record(request, revision, prompt_tokens=gen.prompt_tokens,
                            completion_tokens=gen.completion_tokens, raw_text=gen.text,
                            parsed=parsed, parse_ok=True, attempts=number,
                            finish_reason=gen.finish_reason, flags=flags, attempt_log=history,
                            max_tokens=max_tokens, seed_used=used_seed, **extra)
                )
                summary["ok"] += 1
            store.append(records)
            live = remaining
    return summary


def make_engine(entry, revision):
    return VLLMEngine(entry, revision)
