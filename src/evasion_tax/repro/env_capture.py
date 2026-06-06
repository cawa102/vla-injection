"""Capture the runtime environment for reproducibility logging.

Records platform, Python version, an installed-dependency snapshot, the git
commit, and torch/CUDA/driver versions when present. Every field degrades
gracefully (to ``None``) rather than raising, so the same call works on a
local dev host (no torch, no CUDA) and on the GPU node (A100/H100).
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform as platform_mod
import subprocess
import sys


def _git_commit() -> str | None:
    """Return the current ``HEAD`` commit hash, or ``None`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _dependency_snapshot() -> dict[str, str]:
    """Return a ``{distribution: version}`` snapshot of installed packages.

    Uses ``importlib.metadata`` so no subprocess is required and the result is a
    sortable, JSON-serialisable mapping.
    """
    snapshot: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"]
        if name is None:
            continue
        snapshot[name] = dist.version
    return dict(sorted(snapshot.items(), key=lambda kv: kv[0].lower()))


def _torch_versions() -> tuple[str | None, str | None, str | None]:
    """Return ``(torch_version, cuda_version, driver_version)``.

    All three are ``None`` when torch is not importable (the no-CUDA dev-host case).
    """
    try:
        # Optional/soft import by design: torch is absent on a local dev host without CUDA.
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None, None, None

    torch_version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    driver_version: str | None = None
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.driver_version()  # type: ignore[attr-defined]
            driver_version = f"{major}.{minor}"
    except (AttributeError, RuntimeError, ValueError, TypeError):
        driver_version = None

    return torch_version, cuda_version, driver_version


def capture_env() -> dict:
    """Capture a reproducibility snapshot of the current environment.

    Returns:
        A fresh dict with keys ``platform``, ``python_version``,
        ``dependencies``, ``git_commit``, ``torch``, ``cuda``, ``driver``.
        On a machine without torch/CUDA, ``torch``/``cuda``/``driver`` are
        ``None`` and ``git_commit`` is ``None`` outside a git repo. Never raises.
    """
    torch_version, cuda_version, driver_version = _torch_versions()
    return {
        "platform": platform_mod.platform(),
        "python_version": sys.version.split()[0],
        "dependencies": _dependency_snapshot(),
        "git_commit": _git_commit(),
        "torch": torch_version,
        "cuda": cuda_version,
        "driver": driver_version,
    }
