"""Quarantined suffix artifacts and bf16-transfer records for surrogate GCG.

These records are deliberately model-free. GPU scripts create them around the
existing OpenVLA/GCG seam; tests can round-trip and validate provenance locally.
Suffix text/token artifacts live under ``artifacts/untrusted/`` by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from evasion_tax.attack.openvla_loader import OpenVlaPrecision

StrPath = str | PathLike[str]
SCHEMA_VERSION = 2
QUARANTINE_ROOT = Path("artifacts/untrusted")
PRECISIONS = ("bf16", "int8", "nf4_4bit")
_REQUIRED_GCG_KEYS = ("suffix_len", "n_steps", "top_k", "search_width", "early_stop")


def utc_now_iso() -> str:
    """UTC timestamp for artifact records."""
    return datetime.now(timezone.utc).isoformat()


def _coerce_int_tuple(values: tuple[int, ...] | list[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return out


def _require_nonempty(value: str | None, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required for suffix provenance")
    return value


def _sha256_is_plausible(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def is_under_quarantine(path: StrPath, quarantine_root: StrPath = QUARANTINE_ROOT) -> bool:
    """Return whether ``path`` resolves under ``quarantine_root``."""
    root = Path(quarantine_root).resolve()
    candidate = Path(path).resolve()
    return candidate == root or candidate.is_relative_to(root)


def require_quarantined(path: StrPath, quarantine_root: StrPath = QUARANTINE_ROOT) -> Path:
    """Return ``path`` as a Path, raising unless it is under ``artifacts/untrusted``."""
    candidate = Path(path)
    if not is_under_quarantine(candidate, quarantine_root):
        raise ValueError(f"artifact path must stay under {quarantine_root}: {candidate}")
    return candidate


@dataclass(frozen=True)
class SurrogateSuffixArtifact:
    """One optimized RoboGCG suffix and its surrogate-run provenance."""

    schema_version: int
    artifact_id: str
    surrogate_precision: OpenVlaPrecision
    model_checkpoint: str
    suite: str
    task_id: str
    target_action_tokens: tuple[int, ...]
    seed: int
    gcg_config: dict[str, Any]
    suffix_token_ids: tuple[int, ...]
    suffix_path: str
    suffix_sha256: str
    source_run_dir: str
    git_commit: str
    gpu_id: str
    environment: dict[str, Any]
    load_record: dict[str, Any]
    surrogate_target_hit: bool
    surrogate_steps_to_success: int
    surrogate_censored: bool
    surrogate_best_loss: float | None
    surrogate_wall_seconds: float
    surrogate_peak_vram_gib: float | None
    surrogate_gradient_health: dict[str, Any] | None
    failure_reason: str | None
    created_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported artifact schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.surrogate_precision not in PRECISIONS:
            raise ValueError(
                f"surrogate_precision must be one of {PRECISIONS}, "
                f"got {self.surrogate_precision!r}"
            )
        for name in (
            "artifact_id",
            "model_checkpoint",
            "suite",
            "task_id",
            "suffix_path",
            "suffix_sha256",
            "source_run_dir",
            "git_commit",
            "gpu_id",
            "created_utc",
        ):
            _require_nonempty(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "target_action_tokens",
            _coerce_int_tuple(self.target_action_tokens, name="target_action_tokens"),
        )
        object.__setattr__(
            self,
            "suffix_token_ids",
            _coerce_int_tuple(self.suffix_token_ids, name="suffix_token_ids"),
        )
        missing = [k for k in _REQUIRED_GCG_KEYS if k not in self.gcg_config]
        if missing:
            raise ValueError(f"gcg_config missing required keys: {missing}")
        if not _sha256_is_plausible(self.suffix_sha256):
            raise ValueError("suffix_sha256 must be a 64-character hex digest")
        require_quarantined(self.suffix_path)
        if self.surrogate_steps_to_success < 0:
            raise ValueError("surrogate_steps_to_success must be >= 0")
        if self.surrogate_wall_seconds < 0:
            raise ValueError("surrogate_wall_seconds must be >= 0")
        if self.surrogate_peak_vram_gib is not None and self.surrogate_peak_vram_gib < 0:
            raise ValueError("surrogate_peak_vram_gib must be >= 0 when provided")
        if self.surrogate_best_loss is not None and not math.isfinite(self.surrogate_best_loss):
            raise ValueError("surrogate_best_loss must be finite when provided")
        if self.surrogate_gradient_health is not None and not isinstance(
            self.surrogate_gradient_health, dict
        ):
            raise ValueError("surrogate_gradient_health must be a dict or None when provided")

    @property
    def surrogate_gpu_hours(self) -> float:
        """Surrogate optimization wall time in GPU-hours."""
        return self.surrogate_wall_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        obj = asdict(self)
        obj["target_action_tokens"] = list(self.target_action_tokens)
        obj["suffix_token_ids"] = list(self.suffix_token_ids)
        return obj

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> SurrogateSuffixArtifact:
        """Build and validate from a parsed JSON mapping."""
        data = dict(obj)
        data["target_action_tokens"] = tuple(data["target_action_tokens"])
        data["suffix_token_ids"] = tuple(data["suffix_token_ids"])
        return cls(**data)


@dataclass(frozen=True)
class TransferEvalRecord:
    """bf16 victim re-evaluation of one surrogate suffix artifact."""

    schema_version: int
    transfer_id: str
    artifact_id: str
    artifact_path: str
    suffix_sha256: str
    surrogate_precision: OpenVlaPrecision
    surrogate_target_hit: bool
    victim_precision: str
    model_checkpoint: str
    suite: str
    task_id: str
    seed: int
    target_action_tokens: tuple[int, ...]
    victim_target_hit: bool
    predicted_target_tokens: tuple[int, ...] | None
    action_distance_to_target: float | None
    persistence_window: int
    rollout_evaluated: bool
    rollout_success: bool | None
    wall_seconds: float
    failure_reason: str | None
    censored: bool
    source_run_dir: str
    git_commit: str
    environment: dict[str, Any]
    created_utc: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported transfer schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.surrogate_precision not in PRECISIONS:
            raise ValueError(
                f"surrogate_precision must be one of {PRECISIONS}, "
                f"got {self.surrogate_precision!r}"
            )
        if self.victim_precision != "bf16":
            raise ValueError("victim_precision must be 'bf16'")
        for name in (
            "transfer_id",
            "artifact_id",
            "artifact_path",
            "suffix_sha256",
            "model_checkpoint",
            "suite",
            "task_id",
            "source_run_dir",
            "git_commit",
            "created_utc",
        ):
            _require_nonempty(getattr(self, name), name=name)
        if not _sha256_is_plausible(self.suffix_sha256):
            raise ValueError("suffix_sha256 must be a 64-character hex digest")
        object.__setattr__(
            self,
            "target_action_tokens",
            _coerce_int_tuple(self.target_action_tokens, name="target_action_tokens"),
        )
        if self.predicted_target_tokens is not None:
            object.__setattr__(
                self,
                "predicted_target_tokens",
                _coerce_int_tuple(self.predicted_target_tokens, name="predicted_target_tokens"),
            )
        if self.persistence_window < 1:
            raise ValueError("persistence_window must be >= 1")
        if self.wall_seconds < 0:
            raise ValueError("wall_seconds must be >= 0")
        if (
            self.action_distance_to_target is not None
            and not math.isfinite(self.action_distance_to_target)
        ):
            raise ValueError("action_distance_to_target must be finite when provided")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        obj = asdict(self)
        obj["target_action_tokens"] = list(self.target_action_tokens)
        if self.predicted_target_tokens is not None:
            obj["predicted_target_tokens"] = list(self.predicted_target_tokens)
        return obj

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> TransferEvalRecord:
        """Build and validate from a parsed JSON mapping."""
        data = dict(obj)
        data["target_action_tokens"] = tuple(data["target_action_tokens"])
        if data.get("predicted_target_tokens") is not None:
            data["predicted_target_tokens"] = tuple(data["predicted_target_tokens"])
        return cls(**data)


def write_json_record(record: SurrogateSuffixArtifact | TransferEvalRecord, path: StrPath) -> Path:
    """Write a record JSON once; raise if the file already exists."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact record: {target}")
    target.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    return target


def read_suffix_artifact(path: StrPath) -> SurrogateSuffixArtifact:
    """Read and validate a quarantined suffix artifact JSON."""
    require_quarantined(path)
    return SurrogateSuffixArtifact.from_dict(json.loads(Path(path).read_text()))


def read_transfer_record(path: StrPath) -> TransferEvalRecord:
    """Read and validate a transfer-eval record JSON."""
    return TransferEvalRecord.from_dict(json.loads(Path(path).read_text()))


def token_l2_distance(
    predicted: tuple[int, ...] | list[int],
    target: tuple[int, ...] | list[int],
) -> float:
    """Euclidean distance between predicted and target action-token ids."""
    pred = _coerce_int_tuple(tuple(predicted), name="predicted")
    tgt = _coerce_int_tuple(tuple(target), name="target")
    if len(pred) != len(tgt):
        raise ValueError(f"predicted and target token lengths differ: {len(pred)} != {len(tgt)}")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pred, tgt, strict=True)))


def compatibility_mismatches(
    artifact: SurrogateSuffixArtifact,
    *,
    model_checkpoint: str,
    suite: str,
    task_id: str,
) -> list[str]:
    """Return checkpoint/suite/task mismatch labels for a victim transfer run."""
    mismatches: list[str] = []
    if artifact.model_checkpoint != model_checkpoint:
        mismatches.append("model_checkpoint")
    if artifact.suite != suite:
        mismatches.append("suite")
    if artifact.task_id != task_id:
        mismatches.append("task_id")
    return mismatches


def assert_transfer_compatible(
    artifact: SurrogateSuffixArtifact,
    *,
    model_checkpoint: str,
    suite: str,
    task_id: str,
    override: bool = False,
) -> None:
    """Reject incompatible artifact/victim identities unless explicitly overridden."""
    mismatches = compatibility_mismatches(
        artifact,
        model_checkpoint=model_checkpoint,
        suite=suite,
        task_id=task_id,
    )
    if mismatches and not override:
        raise ValueError(
            "suffix artifact is incompatible with victim run "
            f"({', '.join(mismatches)} mismatch); pass an explicit override to continue"
        )
