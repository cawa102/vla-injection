---
source_file: "scripts/run_benign.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# run_benign.py

## Connections
- [[main()_10]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`scripts/run_benign.py`

```python
#!/usr/bin/env python3
"""Run the benign LIBERO baseline for a pinned config (GPU-node-only).

Validates the config locally, then **guards**: OpenVLA-7B + CUDA do not exist on
the 8 GB host, so with no CUDA runtime this prints the GPU-node requirement and exits
non-zero (never a silent no-op). The benign-rollout body is implemented on the GPU
node (see ``docs/setup/gpu-runbook.md``); it stands up the bf16 policy on LIBERO and
logs each run via ``RunLogger`` to write-once ``results/``.

Usage:
    python scripts/run_benign.py --config configs/example_m2.yaml
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.config import cuda_available, gpu_required_message, load_config  # noqa: E402

STAGE = "run_benign"
_EXIT_REQUIRES_GPU = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    args = parser.parse_args(argv)

    # Validate the config now so errors surface locally, before GPU-node time.
    load_config(args.config)

    if not cuda_available():
        print(gpu_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GPU

    raise NotImplementedError(
        "GPU: benign-rollout execution (OpenVLA-7B + LIBERO) is not available "
        "locally; implement against the GPU runbook (docs/setup/gpu-runbook.md)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

