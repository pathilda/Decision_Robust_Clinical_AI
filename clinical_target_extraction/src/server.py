"""Lifecycle management for a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from tqdm.auto import tqdm

from .settings import ModelPreset


class VLLMServerError(RuntimeError):
    pass


def build_server_command(
    *,
    preset: ModelPreset,
    model_path: Path,
    host: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        "extractor",
        "--dtype",
        preset.dtype,
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        "1",
        "--host",
        host,
        "--port",
        str(port),
    ]


def server_models(host: str, port: int, *, timeout: float = 2.0) -> dict | None:
    try:
        with urlopen(f"http://{host}:{port}/v1/models", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None


class ManagedVLLMServer:
    """Start vLLM with inherited output so native loading progress stays visible."""

    def __init__(
        self,
        *,
        preset: ModelPreset,
        model_path: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.92,
        startup_timeout: int = 1800,
    ) -> None:
        self.preset = preset
        self.model_path = model_path
        self.host = host
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        if server_models(self.host, self.port) is not None:
            raise VLLMServerError(
                f"Port {self.port} already has a vLLM-compatible server. "
                "Stop it first or choose another --port."
            )
        command = build_server_command(
            preset=self.preset,
            model_path=self.model_path,
            host=self.host,
            port=self.port,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        kwargs: dict = {"env": env}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        print(f"\n>>> Loading {self.preset.name} from {self.model_path}", flush=True)
        print(">>> vLLM output and model-loading progress follow:\n", flush=True)
        self.process = subprocess.Popen(command, **kwargs)

    def wait_until_ready(self) -> None:
        if self.process is None:
            raise VLLMServerError("Server has not been started")
        started = time.monotonic()
        with tqdm(
            total=self.startup_timeout,
            desc=f"Waiting for {self.preset.name}",
            unit="s",
            dynamic_ncols=True,
        ) as progress:
            previous = 0
            while True:
                return_code = self.process.poll()
                if return_code is not None:
                    raise VLLMServerError(
                        f"vLLM exited during startup with code {return_code}"
                    )
                if server_models(self.host, self.port) is not None:
                    progress.set_description(f"{self.preset.name} ready")
                    return
                elapsed = int(time.monotonic() - started)
                progress.update(max(0, elapsed - previous))
                previous = elapsed
                if elapsed >= self.startup_timeout:
                    raise VLLMServerError(
                        f"vLLM did not become ready within {self.startup_timeout} seconds"
                    )
                time.sleep(2)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        print(f"\n>>> Stopping {self.preset.name} server...", flush=True)
        if os.name == "nt":
            self.process.terminate()
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                self.process.kill()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=10)

    def __enter__(self) -> "ManagedVLLMServer":
        self.start()
        try:
            self.wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
