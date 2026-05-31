"""Fetch + provenance helpers for OpenVLA ``dataset_statistics.json`` (Task 3).

The action codec needs the per-dataset ``q01``/``q99``/``mask`` statistics that
OpenVLA ships in ``dataset_statistics.json`` on each fine-tuned checkpoint repo.
This module holds the *pure*, network-free logic (artifact URL + provenance
recording); the actual download lives in ``scripts/fetch_openvla_stats.py`` and
runs online / on GB10. Statistics land under the gitignored ``data/`` tree;
their provenance (source, hash, date, licence) is recorded per invariant #8.
"""

from __future__ import annotations

from os import PathLike

from t7.repro.provenance import record_provenance, sha256_file

StrPath = str | PathLike[str]

STATS_FILENAME = "dataset_statistics.json"


def stats_url(repo_id: str) -> str:
    """Return the direct download URL of a checkpoint's ``dataset_statistics.json``.

    Args:
        repo_id: HF repo id, e.g. ``openvla/openvla-7b-finetuned-libero-spatial``.

    Returns:
        The ``.../resolve/main/dataset_statistics.json`` artifact URL.
    """
    return f"https://huggingface.co/{repo_id}/resolve/main/{STATS_FILENAME}"


def record_stats_provenance(
    manifest_path: StrPath,
    *,
    repo_id: str,
    stats_path: StrPath,
    date: str,
    licence: str,
) -> str:
    """Hash a downloaded stats file and record one provenance entry for it.

    Args:
        manifest_path: JSON provenance manifest (created/updated in place).
        repo_id: HF repo the stats were fetched from.
        stats_path: Path to the downloaded ``dataset_statistics.json``.
        date: Retrieval date (ISO ``YYYY-MM-DD``).
        licence: Licence string (pass a ``[VERIFY ...]`` placeholder if unknown —
            do not assert a licence from memory).

    Returns:
        The hex SHA-256 of the stats file (also stored in the manifest).
    """
    sha = sha256_file(stats_path)
    record_provenance(
        manifest_path,
        name=f"openvla-stats:{repo_id}",
        source=stats_url(repo_id),
        sha256=sha,
        date=date,
        licence=licence,
    )
    return sha
