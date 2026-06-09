---
source_file: "scripts/run_attack.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# run_attack.py

## Connections
- [[main()_9]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`scripts/run_attack.py`

```python
#!/usr/bin/env python3
"""Reproduce the RoboGCG attack for a pinned config (GPU-node-only).

Validates the config locally, then **guards**: with no CUDA runtime it prints the
GPU-node requirement and exits non-zero (no silent no-op). The attack body is
implemented on the GPU node (see ``docs/setup/gpu-runbook.md``): it optimises the
white-box adversarial suffix, runs the attacked rollouts, and **quarantines every
suffix under ``artifacts/untrusted/``** (ethics invariant) while logging metrics
to write-once ``results/``.

Usage:
    python scripts/run_attack.py --config configs/example_m2.yaml
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402

STAGE = "run_attack"
_EXIT_REQUIRES_GPU = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    args = parser.parse_args(argv)

    load_config(args.config)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    raise NotImplementedError(
        "GPU: RoboGCG suffix optimisation + attacked rollouts are not available "
        "locally; implement against the GPU runbook (docs/setup/gpu-runbook.md; "
        "quarantine suffixes under artifacts/untrusted/)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

