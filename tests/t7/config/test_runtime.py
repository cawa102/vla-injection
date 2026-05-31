"""Tests for the GB10 runtime guard (Task 9).

The model/GPU-dependent scripts (``run_benign``/``run_attack``/``microbench_gcg``)
must refuse to silently no-op when no CUDA model is present: they print the GB10
requirement and exit non-zero. The decision and the message live here so all
three scripts share one tested guard rather than duplicating it.
"""

from t7.config.runtime import cuda_available, gb10_required_message


def test_cuda_unavailable_on_local_host():
    # The local M1 host has no torch/CUDA, so the guard must fire.
    assert cuda_available() is False


def test_gb10_message_names_the_stage_and_gb10():
    msg = gb10_required_message("run_benign")
    assert "run_benign" in msg
    assert "GB10" in msg
