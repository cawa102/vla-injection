"""Write-once run logger — the gatekeeper for the write-once invariant (#5).

Every run writes to ``results_root/<UTC-timestamp>-<slug>/`` and the logger
*refuses* to overwrite: a colliding run directory raises, and so does writing a
file that already exists. The UTC clock is injectable so tests can pin a
deterministic timestamp.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

import numpy as np

from t7.repro.env_capture import capture_env

StrPath = str | PathLike[str]


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``."""
    return datetime.now(timezone.utc)


def _timestamp_slug(now: datetime) -> str:
    """Format a UTC datetime as a filesystem-safe ``YYYY-MM-DDTHH-MM-SSZ`` token.

    Colons are replaced by hyphens so the stamp is a legal path component on
    every filesystem, while still being unambiguously UTC (trailing ``Z``).
    """
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


class RunHandle:
    """A live handle to one run directory; all writes refuse to overwrite."""

    def __init__(self, run_dir: Path) -> None:
        self._dir = run_dir

    @property
    def dir(self) -> Path:
        """The run's output directory."""
        return self._dir

    def write(self, name: str, obj: object) -> Path:
        """Write ``obj`` as ``<name>.json``; raise if the file already exists.

        Args:
            name: Base filename (no extension).
            obj: Any JSON-serialisable object.

        Returns:
            Path to the written file.

        Raises:
            FileExistsError: If ``<name>.json`` already exists (write-once).
        """
        target = self._dir / f"{name}.json"
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing run artefact: {target} (write-once)"
            )
        target.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return target

    def write_array(self, name: str, arr: np.ndarray) -> Path:
        """Save a numpy array as ``<name>.npy``; raise if the file already exists.

        Args:
            name: Base filename (no extension).
            arr: The array to persist.

        Returns:
            Path to the written ``.npy`` file.

        Raises:
            FileExistsError: If ``<name>.npy`` already exists (write-once).
        """
        target = self._dir / f"{name}.npy"
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing run array: {target} (write-once)"
            )
        np.save(target, np.asarray(arr))
        return target


class RunLogger:
    """Create write-once, UTC-timestamped run directories under ``results_root``.

    Args:
        results_root: Directory under which run folders are created. In tests
            this is always a ``tmp_path``, never the real ``results/``.
        now: Callable returning a timezone-aware UTC ``datetime``. Injectable so
            tests can pin a deterministic timestamp; defaults to real UTC now.
    """

    def __init__(
        self,
        results_root: StrPath,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._root = Path(results_root)
        self._now = now

    def start(self, slug: str, config: dict, seed: int) -> RunHandle:
        """Create a fresh run directory and write its ``run.json`` protocol block.

        Args:
            slug: Short human-readable run name (appended to the timestamp).
            config: The pinned config snapshot; copied, never mutated.
            seed: The pinned seed for this run.

        Returns:
            A :class:`RunHandle` bound to the new directory.

        Raises:
            FileExistsError: If the target run directory already exists
                (write-once invariant).
        """
        stamp = _timestamp_slug(self._now())
        run_id = f"{stamp}-{slug}"
        run_dir = self._root / run_id

        # mkdir(exist_ok=False) is the atomic write-once guard.
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to reuse existing run directory: {run_dir} (write-once)"
            ) from exc

        env = capture_env()
        run_record = {
            "run_id": run_id,
            "git_commit": env["git_commit"],
            "hardware": env,
            "config": copy.deepcopy(config),
            "seed": seed,
            "created_utc": self._now().astimezone(timezone.utc).isoformat(),
            # Protocol placeholders (Playbook §8) to fill after the run.
            "hypothesis": None,
            "expected": None,
            "observed": None,
            "decision": None,
            "one_variable": None,
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True) + "\n"
        )
        return RunHandle(run_dir)
