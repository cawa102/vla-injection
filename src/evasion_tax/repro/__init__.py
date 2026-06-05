"""Reproducibility infrastructure for the Embodiment Evasion Tax project.

The non-negotiable reproducibility layer (CLAUDE.md): deterministic seeding,
environment capture, data/checkpoint provenance, and a write-once run logger.
Pure Python + NumPy; torch is soft-imported and absent on the local M1 machine.
"""

from evasion_tax.repro.env_capture import capture_env
from evasion_tax.repro.provenance import record_provenance, sha256_file
from evasion_tax.repro.run_logger import RunHandle, RunLogger
from evasion_tax.repro.seeds import seed_everything

__all__ = [
    "RunHandle",
    "RunLogger",
    "capture_env",
    "record_provenance",
    "seed_everything",
    "sha256_file",
]
