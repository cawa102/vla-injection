"""Reproducibility infrastructure for T7.

The non-negotiable reproducibility layer (CLAUDE.md): deterministic seeding,
environment capture, data/checkpoint provenance, and a write-once run logger.
Pure Python + NumPy; torch is soft-imported and absent on the local M1 machine.
"""

from t7.repro.env_capture import capture_env
from t7.repro.provenance import record_provenance, sha256_file
from t7.repro.run_logger import RunHandle, RunLogger
from t7.repro.seeds import seed_everything

__all__ = [
    "RunHandle",
    "RunLogger",
    "capture_env",
    "record_provenance",
    "seed_everything",
    "sha256_file",
]
