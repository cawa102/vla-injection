---
source_file: "scripts/calibrate.py"
type: "code"
community: "Config Schema & Immutability"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Config_Schema__Immutability
---

# calibrate.py

## Connections
- [[main()]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Config_Schema__Immutability

## 📄 Source

`scripts/calibrate.py`

```python
#!/usr/bin/env python3
"""Calibrate the detector threshold from logged benign scores (model-free, local).

Sets ``tau`` on a benign calibration split to a per-rollout false-abort budget
via the shared :func:`evasion_tax.detector.calibrate` (the same primitive baselines reuse;
plan invariant #4). Consumes a JSON of per-rollout score arrays — no model needed
— so it runs locally on logged data.

Usage:
    python scripts/calibrate.py --benign-scores benign_calib.json --fpr 0.05

``benign_calib.json`` is a list of rollouts, each a list of per-step scores in
``[0, 1]``: ``[[0.1, 0.2, ...], [0.0, 0.3, ...], ...]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.detector import calibrate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign-scores", required=True, help="benign calib scores JSON")
    parser.add_argument("--fpr", type=float, required=True, help="target per-rollout FPR")
    args = parser.parse_args(argv)

    benign = json.loads(Path(args.benign_scores).read_text())
    threshold = calibrate(benign, target_per_rollout_fpr=args.fpr)

    print(
        json.dumps(
            {
                "tau": threshold.tau,
                "aggregate": threshold.aggregate,
                "target_fpr": threshold.target_fpr,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

