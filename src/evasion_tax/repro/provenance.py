"""Data/checkpoint provenance: SHA-256 hashing and a JSON manifest.

Records the source, hash, date, and licence of every fetched artefact (the
fields mirrored in ``docs/references/README.md``). The manifest is keyed by
``name`` so re-recording the same artefact updates *only* that entry, never the
whole file — keeping provenance append-safe across runs.
"""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB streaming chunk so large artefacts don't load into memory.

StrPath = str | PathLike[str]


def sha256_file(path: StrPath) -> str:
    """Return the hex SHA-256 digest of a file, read in streaming chunks.

    Args:
        path: Path to an existing file.

    Returns:
        The 64-char lowercase hex digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_provenance(
    manifest_path: StrPath,
    *,
    name: str,
    source: str,
    sha256: str,
    date: str,
    licence: str,
) -> None:
    """Append or update one provenance entry in a JSON manifest.

    The manifest is a JSON object keyed by ``name``. If it does not exist it is
    created; if an entry with the same ``name`` exists it is overwritten while
    every other entry is preserved.

    Args:
        manifest_path: Path to the JSON manifest (created if missing).
        name: Unique key for the artefact (one entry per name).
        source: Origin URL or description.
        sha256: Hex digest of the artefact.
        date: Retrieval date (ISO ``YYYY-MM-DD``).
        licence: Licence string.
    """
    path = Path(manifest_path)

    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Provenance manifest at {path} is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Provenance manifest at {path} must be a JSON object")
    else:
        manifest = {}

    entry = {
        "name": name,
        "source": source,
        "sha256": sha256,
        "date": date,
        "licence": licence,
    }
    # Build a new mapping (immutability) rather than mutating the loaded dict.
    updated = {**manifest, name: entry}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
