#!/usr/bin/env python3
"""GCG attack micro-benchmark — resolves D4 (eval scale) and D7 (compute budget).

Validates the config locally, then **guards**: with no CUDA runtime it prints the
GB10 requirement and exits non-zero (no silent no-op). The benchmark body is
implemented on GB10 (see ``docs/setup/gb10-runbook.md``): it measures GCG
seconds-per-target at 4-bit on a few tasks so the M1 gate can fix the final eval
matrix and compute budget, logging the timings to write-once ``results/``.

Usage:
    python scripts/microbench_gcg.py --config configs/example_m2.yaml
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from t7.config import cuda_available, gb10_required_message, load_config  # noqa: E402

STAGE = "microbench_gcg"
_EXIT_REQUIRES_GB10 = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="pinned config YAML")
    parser.add_argument("--results-root", default="results", help="write-once results root")
    args = parser.parse_args(argv)

    load_config(args.config)

    if not cuda_available():
        print(gb10_required_message(STAGE), file=sys.stderr)
        return _EXIT_REQUIRES_GB10

    raise NotImplementedError(
        "GB10: GCG seconds-per-target benchmarking is not available locally; "
        "implement against the GB10 runbook."
    )


if __name__ == "__main__":
    raise SystemExit(main())
