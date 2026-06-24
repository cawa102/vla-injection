#!/usr/bin/env python3
"""Summarize logged surrogate-transfer records into ASR tables and figures."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import _bootstrap  # noqa: F401  (import side effect: puts src/ on sys.path)

from evasion_tax.attack.surrogate_artifacts import read_transfer_record  # noqa: E402
from evasion_tax.eval.surrogate_transfer import (  # noqa: E402
    summarize_transfer,
    write_summary_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="transfer_eval.json paths or glob patterns",
    )
    parser.add_argument("--out-dir", required=True, help="new write-once summary directory")
    return parser


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = [read_transfer_record(p) for p in _expand(args.records)]
    summary = summarize_transfer(records)
    for path in write_summary_outputs(summary, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
