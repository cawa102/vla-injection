"""Tests for the GPU-node runtime guard (Task 9).

The model/GPU-dependent scripts (``run_benign``/``run_attack``/``microbench_gcg``)
must refuse to silently no-op when no CUDA model is present: they print the GPU-node
requirement and exit non-zero. The decision and the message live here so all
three scripts share one tested guard rather than duplicating it.
"""

from t7.config.runtime import cuda_available, gpu_required_message


def test_cuda_unavailable_on_local_host():
    # The local M1 host has no torch/CUDA, so the guard must fire.
    assert cuda_available() is False


def test_gpu_message_names_the_stage_and_gpu():
    msg = gpu_required_message("run_benign")
    assert "run_benign" in msg
    assert "GPU" in msg
