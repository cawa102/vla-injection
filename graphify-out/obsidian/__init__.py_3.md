---
source_file: "src/evasion_tax/config/__init__.py"
type: "code"
community: "Pinned-config schema + runtime guard (Ta"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Pinned-config_schema__runtime_guard_Ta
---

# __init__.py

## Connections
- [[Pinned-config schema + runtime guard (Task 9).  The boundary layer for a run's p]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Pinned-config_schema__runtime_guard_Ta

## 📄 Source

`src/evasion_tax/config/__init__.py`

```python
"""Pinned-config schema + runtime guard (Task 9).

The boundary layer for a run's parameters and host capability:

* :class:`Config` and :func:`load_config` — validate a YAML config (frozen,
  unknown-field-rejecting); :func:`one_variable_diff` reports the leaf paths that
  differ between two configs (the one-variable-at-a-time run discipline).
* :func:`cuda_available` / :func:`gpu_required_message` in
  :mod:`evasion_tax.config.runtime` — the shared GPU-node guard the model-dependent scripts
  use to refuse to silently no-op on the local host.
"""

from __future__ import annotations

from evasion_tax.config.runtime import cuda_available, gpu_required_message
from evasion_tax.config.schema import (
    AttackConfig,
    Config,
    DetectorConfig,
    EnvConfig,
    EvalConfig,
    MetricConfig,
    ModelConfig,
    SplitManifest,
    SplitsConfig,
    load_config,
    one_variable_diff,
)

__all__ = [
    "AttackConfig",
    "Config",
    "DetectorConfig",
    "EnvConfig",
    "EvalConfig",
    "MetricConfig",
    "ModelConfig",
    "SplitManifest",
    "SplitsConfig",
    "cuda_available",
    "gpu_required_message",
    "load_config",
    "one_variable_diff",
]
```

