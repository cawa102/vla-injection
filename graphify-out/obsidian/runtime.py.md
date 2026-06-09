---
source_file: "src/evasion_tax/config/runtime.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# runtime.py

## Connections
- [[GPU-node runtime guard for the modelGPU-dependent scripts (Task 9).  ``run_beni]] - `rationale_for` [EXTRACTED]
- [[Return ``True`` iff a CUDA-capable torch runtime is present.      On a local dev]] - `defined_in` [EXTRACTED]
- [[Return the requires GPU node message printed when the guard fires.      Args]] - `defined_in` [EXTRACTED]
- [[cuda_available()]] - `contains` [EXTRACTED]
- [[gpu_required_message()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`src/evasion_tax/config/runtime.py`

```python
"""GPU-node runtime guard for the model/GPU-dependent scripts (Task 9).

``run_benign`` / ``run_attack`` / ``microbench_gcg`` need OpenVLA-7B + CUDA, which
do not exist on a local dev host without CUDA. Rather than silently no-op, each script
calls :func:`cuda_available` and, when it returns ``False``, prints
:func:`gpu_required_message` and exits non-zero. Keeping both here means the
three scripts share one tested guard instead of duplicating the check.
"""

from __future__ import annotations


def cuda_available() -> bool:
    """Return ``True`` iff a CUDA-capable torch runtime is present.

    On a local dev host without a CUDA torch build this returns ``False``
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


def gpu_required_message(stage: str) -> str:
    """Return the "requires GPU node" message printed when the guard fires.

    Args:
        stage: The script/stage name (e.g. ``"run_benign"``) — surfaced so the
            user sees which step needs the GPU node.
    """
    return (
        f"{stage}: requires the GPU node (A100/H100; OpenVLA-7B + CUDA). No CUDA "
        "runtime is available on this host, so this step cannot run locally. See "
        "docs/setup/gpu-runbook.md."
    )
```

