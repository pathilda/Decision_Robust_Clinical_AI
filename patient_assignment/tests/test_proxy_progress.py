import io
from types import SimpleNamespace

import pytest

from patient_assignment.proxy_progress import Progress, computation_bytes, solver_iteration


def test_display_blocks_preserve_original_bytes():
    original = b"x = 1\r\ny = 2\r\n"
    marked = b"x = 1\r\n# progress-only: begin\r\nprint(1)\r\n# progress-only: end\r\ny = 2\r\n"
    assert computation_bytes(marked) == original
    assert computation_bytes(marked.replace(b"x = 1", b"x = 3")) != original


@pytest.mark.parametrize(
    "raw",
    [
        b"# progress-only: begin\n",
        b"# progress-only: end\n",
        b"# progress-only: begin\n# progress-only: begin\n",
    ],
)
def test_malformed_display_blocks_rejected(raw):
    with pytest.raises(ValueError):
        computation_bytes(raw)


def test_counts_context_and_cache_display():
    output = io.StringIO()
    with Progress(2, SimpleNamespace(hits=4, misses=1), file=output) as progress:
        progress.task(
            {"chunk_id": "01", "profile_id": "balanced"},
            "Qwen",
            "selection",
            {"policy": "robust", "gamma_fraction": 0.25, "probability_threshold": 0.05},
            42,
        )
        solver_iteration("robust", 3)
        progress.advance(True)
        progress.advance(False)
    assert progress.bar.n == 2
    assert progress.failed == 1
    assert "robust master iteration 3" in output.getvalue()
    assert "cached=4, new=1, failed=1" in output.getvalue()
    assert "Gamma=0.25" in output.getvalue()
    assert not progress._thread.is_alive()
    solver_iteration("robust", 4)  # Context has been reset after closing.


def test_disabled_progress_is_silent():
    output = io.StringIO()
    with Progress(1, SimpleNamespace(hits=0, misses=0), disabled=True, file=output) as progress:
        progress.phase("solving")
        progress.advance(True)
    assert output.getvalue() == ""
    assert progress._thread is None


def test_interrupt_closes_bar_and_preserves_partial_count():
    output = io.StringIO()
    with (
        pytest.raises(KeyboardInterrupt),
        Progress(2, SimpleNamespace(hits=0, misses=1), file=output) as progress,
    ):
        progress.advance(True)
        raise KeyboardInterrupt()
    assert progress.bar.n == 1
    assert "interrupted" in output.getvalue()
    assert not progress._thread.is_alive()
