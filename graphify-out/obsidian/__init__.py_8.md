---
source_file: "src/evasion_tax/repro/__init__.py"
type: "code"
community: "Reproducibility infrastructure for the E"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Reproducibility_infrastructure_for_the_E
---

# __init__.py

## Connections
- [[Reproducibility infrastructure for the Embodiment Evasion Tax project.  The non-]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Reproducibility_infrastructure_for_the_E

## 📄 Source

`src/evasion_tax/repro/__init__.py`

```python
"""Reproducibility infrastructure for the Embodiment Evasion Tax project.

The non-negotiable reproducibility layer (CLAUDE.md): deterministic seeding,
environment capture, data/checkpoint provenance, and a write-once run logger.
Pure Python + NumPy; torch is soft-imported and absent on a local dev host without CUDA.
"""

from evasion_tax.repro.env_capture import capture_env
from evasion_tax.repro.provenance import record_provenance, sha256_file
from evasion_tax.repro.run_logger import RunHandle, RunLogger
from evasion_tax.repro.seeds import seed_everything, stable_seed

__all__ = [
    "RunHandle",
    "RunLogger",
    "capture_env",
    "record_provenance",
    "seed_everything",
    "sha256_file",
    "stable_seed",
]
```

