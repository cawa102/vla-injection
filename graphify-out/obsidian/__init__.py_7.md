---
source_file: "src/evasion_tax/policy/__init__.py"
type: "code"
community: "Policy-side helpers (Task 3).  Currently"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Policy-side_helpers_Task_3__Currently
---

# __init__.py

## Connections
- [[Policy-side helpers (Task 3).  Currently exposes the OpenVLA action codec (discr]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Policy-side_helpers_Task_3__Currently

## 📄 Source

`src/evasion_tax/policy/__init__.py`

```python
"""Policy-side helpers (Task 3).

Currently exposes the OpenVLA action codec (discrete token ids -> continuous
7-DoF). Model inference itself is deferred to the GPU node; this is the model-free decode
path, with its formula verified against the OpenVLA source (see
``action_codec`` module docstring for provenance).
"""

from evasion_tax.policy.action_codec import ActionCodec

__all__ = ["ActionCodec"]
```

