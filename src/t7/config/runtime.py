"""GB10 runtime guard for the model/GPU-dependent scripts (Task 9).

``run_benign`` / ``run_attack`` / ``microbench_gcg`` need OpenVLA-7B + CUDA, which
do not exist on the local 8 GB host. Rather than silently no-op, each script
calls :func:`cuda_available` and, when it returns ``False``, prints
:func:`gb10_required_message` and exits non-zero. Keeping both here means the
three scripts share one tested guard instead of duplicating the check.
"""

from __future__ import annotations


def cuda_available() -> bool:
    """Return ``True`` iff a CUDA-capable torch runtime is present.

    On the local M1 host torch is not installed, so this returns ``False``
    (the guard fires); it never raises.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except (RuntimeError, AttributeError):
        return False


def gb10_required_message(stage: str) -> str:
    """Return the "requires GB10" message printed when the guard fires.

    Args:
        stage: The script/stage name (e.g. ``"run_benign"``) — surfaced so the
            user sees which step needs the GPU node.
    """
    return (
        f"{stage}: requires the GB10 node (OpenVLA-7B + CUDA). No CUDA runtime "
        "is available on this host, so this step cannot run locally. See "
        "docs/setup/gb10-runbook.md."
    )
