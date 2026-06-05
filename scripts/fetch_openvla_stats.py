#!/usr/bin/env python3
"""Download an OpenVLA checkpoint's ``dataset_statistics.json`` + record provenance.

Online / GPU step (needs ``huggingface_hub`` + network). The action codec's
quantiles come from this file; it lands under the gitignored ``data/`` tree and
its provenance (source, hash, date, licence) is recorded per invariant #8.

Usage:
    python scripts/fetch_openvla_stats.py \\
        --repo-id openvla/openvla-7b-finetuned-libero-spatial \\
        [--out-dir data/openvla] [--licence "<from model card>"]

The pure URL/provenance logic lives in ``evasion_tax.policy.openvla_stats`` (unit-tested);
this wrapper only performs the network download.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# Make `src/evasion_tax` importable when this script is run standalone (no editable install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evasion_tax.policy.openvla_stats import (  # noqa: E402
    STATS_FILENAME,
    record_stats_provenance,
    stats_url,
)

DEFAULT_OUT_DIR = "data/openvla"
LICENCE_PLACEHOLDER = "[VERIFY: see HF model card]"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True, help="HF checkpoint repo id")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="download destination dir")
    p.add_argument(
        "--manifest",
        default=None,
        help="provenance manifest path (default: <out-dir>/provenance.json)",
    )
    p.add_argument(
        "--licence",
        default=LICENCE_PLACEHOLDER,
        help="licence string from the model card (do not guess; leave placeholder if unsure)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required to download stats. "
            "Install it (it is a core dependency) and retry.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest) if args.manifest else out_dir / "provenance.json"

    print(f"Downloading {STATS_FILENAME} from {stats_url(args.repo_id)} ...")
    local_path = hf_hub_download(
        repo_id=args.repo_id, filename=STATS_FILENAME, local_dir=str(out_dir)
    )

    date = datetime.date.today().isoformat()
    sha = record_stats_provenance(
        manifest,
        repo_id=args.repo_id,
        stats_path=local_path,
        date=date,
        licence=args.licence,
    )

    print(f"Saved: {local_path}")
    print(f"SHA-256: {sha}")
    print(f"Provenance recorded in: {manifest}")
    if args.licence == LICENCE_PLACEHOLDER:
        print(
            "NOTE: licence left as placeholder — verify the checkpoint's licence on "
            "its HF model card and re-run with --licence, or edit the manifest.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
