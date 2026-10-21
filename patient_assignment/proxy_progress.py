"""Display-only progress, deliberately excluded from optimization cache identity.

Only explicitly marked display blocks are removed when hashing existing source files.
All other bytes retain their original hashes, including solver and validation changes.
Keep mathematical, input, and output behavior out of progress-only blocks.
"""

import hashlib
from contextvars import ContextVar
from pathlib import Path
from threading import Event, RLock, Thread
from time import monotonic

from tqdm import tqdm

from .experiment_inputs import digest

_CURRENT = ContextVar("proxy_progress", default=None)
_BEGIN = b"# progress-only: begin"
_END = b"# progress-only: end"


def computation_bytes(raw):
    """Strip only balanced, explicitly delimited display additions, preserving line endings."""
    result = []
    inside = False
    for line in raw.splitlines(keepends=True):
        if line.strip() == _BEGIN:
            if inside:
                raise ValueError("Nested progress-only block")
            inside = True
        elif line.strip() == _END:
            if not inside:
                raise ValueError("Unmatched progress-only block")
            inside = False
        elif not inside:
            result.append(line)
    if inside:
        raise ValueError("Unclosed progress-only block")
    return b"".join(result)


def implementation_hash():
    """Keep pre-progress checkpoints compatible; changes outside display blocks invalidate."""
    directory = Path(__file__).parent
    return digest(
        {
            p.name: hashlib.sha256(computation_bytes(p.read_bytes())).hexdigest()
            for p in sorted(directory.glob("*.py"))
            if p.name != Path(__file__).name
        }
    )


def announce(args, message):
    if not getattr(args, "no_progress", False) and not getattr(args, "dry_run", False):
        tqdm.write(message)


def solver_iteration(policy, iteration):
    progress = _CURRENT.get()
    if progress is not None:
        progress.phase(f"{policy} master iteration {iteration}")


class Progress:
    """Attempted selections, not solver iterations or percentage of wall-clock work."""

    def __init__(self, total, cache, disabled=False, file=None, interval=2.0):
        self.cache = cache
        self.disabled = disabled
        self.interval = interval
        self.failed = 0
        self.context = ""
        self.stage = "initializing"
        self._stop = Event()
        self._lock = RLock()
        self._last = 0.0
        self._thread = None
        self.bar = tqdm(
            total=total,
            desc="Proxy analysis",
            unit="selection",
            disable=disabled,
            file=file,
            dynamic_ncols=True,
            mininterval=1.0,
            ascii=True,
        )

    def __enter__(self):
        self._token = _CURRENT.set(self)
        if not self.disabled:
            self._thread = Thread(target=self._heartbeat, name="proxy-progress", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if exc_type is not None:
            self.stage = (
                "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "stopped with error"
            )
        self._render(force=True)
        self.bar.close()
        _CURRENT.reset(self._token)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._render(force=True)

    def _render(self, force=False):
        if self.disabled:
            return
        with self._lock:
            now = monotonic()
            if force or now - self._last >= 1.0:
                self.bar.set_postfix_str(
                    f"{self.context}; {self.stage}; cached={self.cache.hits}, "
                    f"new={self.cache.misses}, failed={self.failed}",
                    refresh=False,
                )
                self.bar.refresh()
                self._last = now

    def task(self, instance, model, phase, cell=None, seed=None):
        with self._lock:
            details = ""
            if cell:
                details = cell["policy"]
                if cell["policy"] == "stochastic":
                    details += f" W={cell['scenario_count']} r={cell['scenario_seed']} T={cell['temperature']}"
                elif cell["policy"] == "robust":
                    details += f" Gamma={cell.get('gamma_fraction', cell.get('gamma_count'))} tau={cell['probability_threshold']}"
            self.context = f"{model} {details} | {instance['chunk_id']} / {instance['profile_id']}"
            if seed is not None:
                self.context += f" | selection seed={seed}"
            self.stage = phase
        self._render()

    def phase(self, stage):
        with self._lock:
            self.stage = stage
        self._render()

    def advance(self, success):
        with self._lock:
            self.failed += not success
            self.bar.update(1)
        self._render()
