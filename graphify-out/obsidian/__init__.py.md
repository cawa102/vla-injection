---
source_file: "src/evasion_tax/__init__.py"
type: "code"
community: "Package Smoke Test"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Package_Smoke_Test
---

# __init__.py

## Connections
- [[Embodiment Evasion Tax — goal-action consistency detection for instruction-injec]] - `rationale_for` [EXTRACTED]
- [[evasion_tax package]] - `defined_in` [EXTRACTED]
- [[evasion_tax package (model-free core)]] - `defined_in` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Package_Smoke_Test

## 📄 Source

`src/evasion_tax/__init__.py`

```python
"""Embodiment Evasion Tax — goal-action consistency detection for instruction-injected VLA policies.

This package holds the **model-free** components that can be built and unit-tested on a
local dev host without CUDA: reproducibility infrastructure, the privileged-state consistency
metric (A), the FP-calibrated detector, evaluation statistics, the OpenVLA action codec,
and baselines.

Model/GPU-dependent pieces (OpenVLA-7B inference, GCG optimisation, LIBERO *rollouts*) are
deferred to the GPU node (A100/H100) and sit behind thin interfaces with synthetic fixtures
for tests.

See ``docs/core/local-prep-plan.md`` for the task breakdown.
"""

__version__ = "0.0.0"
```

