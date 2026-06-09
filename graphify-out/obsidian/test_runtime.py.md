---
source_file: "tests/evasion_tax/config/test_runtime.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# test_runtime.py

## Connections
- [[Tests for the GPU-node runtime guard (Task 9).  The modelGPU-dependent scripts]] - `rationale_for` [EXTRACTED]
- [[cuda_available()]] - `references` [EXTRACTED]
- [[gpu_required_message()]] - `references` [EXTRACTED]
- [[test_cuda_unavailable_on_local_host()]] - `contains` [EXTRACTED]
- [[test_gpu_message_names_the_stage_and_gpu()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`tests/evasion_tax/config/test_runtime.py`

```python
"""Tests for the GPU-node runtime guard (Task 9).

The model/GPU-dependent scripts (``run_benign``/``run_attack``/``microbench_gcg``)
must refuse to silently no-op when no CUDA model is present: they print the GPU-node
requirement and exit non-zero. The decision and the message live here so all
three scripts share one tested guard rather than duplicating it.
"""

from evasion_tax.config.runtime import cuda_available, gpu_required_message


def test_cuda_unavailable_on_local_host():
    # The local M1 host has no torch/CUDA, so the guard must fire.
    assert cuda_available() is False


def test_gpu_message_names_the_stage_and_gpu():
    msg = gpu_required_message("run_benign")
    assert "run_benign" in msg
    assert "GPU" in msg
```

