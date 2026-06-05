"""Pinned-config schema + one-variable-diff (Task 9).

A run's parameters are validated at the system boundary before anything executes
(coding-style: never trust external data; fail fast with a clear message). Every
model is **frozen** (plan invariant #6) and **forbids unknown fields** so a typo
like ``window`` for ``k`` is rejected rather than silently ignored.

``one_variable_diff`` realises the "change exactly one variable per run"
discipline (Playbook §8): it returns the dotted paths of the *leaf* fields that
differ between two configs, so a run can assert it changed exactly one thing
versus its predecessor.
"""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StrPath = str | PathLike[str]


class _Frozen(BaseModel):
    """Base for every config section: immutable + rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Frozen):
    """The victim policy checkpoint and its action-decoding key."""

    name: str
    unnorm_key: str
    checkpoint: str | None = None
    quantization: str | None = None


class EnvConfig(_Frozen):
    """The LIBERO suite + the tasks/episode length a run covers."""

    suite: str
    tasks: list[str] = Field(min_length=1)
    max_steps: int = Field(gt=0)


class AttackConfig(_Frozen):
    """The attack arm and its window-scored target budget (decision D2)."""

    name: str
    targets_per_task: int = Field(gt=0)
    persistence_steps: int = Field(ge=1)


class MetricConfig(_Frozen):
    """Consistency metric (A) settings — the causal prefix-window length ``k``."""

    k: int = Field(ge=1)


class DetectorConfig(_Frozen):
    """FP-calibration targets — the per-rollout benign false-abort budgets.

    ``primary_fpr`` pins the **pre-registered primary operating point** (Codex
    review #2 #3): 5% is the headline; a tighter point is exploratory unless its
    held-out benign N clears the rule-of-three floor (see :mod:`evasion_tax.eval.power`).
    It must be one of ``fpr_targets``.
    """

    fpr_targets: list[float] = Field(min_length=1)
    primary_fpr: float = 0.05

    @field_validator("fpr_targets")
    @classmethod
    def _within_unit_interval(cls, targets: list[float]) -> list[float]:
        for t in targets:
            if not (0.0 < t < 1.0):
                raise ValueError(f"fpr_targets must each be in (0, 1), got {t}")
        return targets

    @model_validator(mode="after")
    def _primary_fpr_is_a_target(self) -> DetectorConfig:
        if not any(math.isclose(self.primary_fpr, t) for t in self.fpr_targets):
            raise ValueError(
                f"primary_fpr {self.primary_fpr} must be one of fpr_targets "
                f"{self.fpr_targets}"
            )
        return self


class SplitManifest(_Frozen):
    """The ids on each disjointness axis for one split (calib or test)."""

    tasks: list[str]
    scenes: list[str]
    seeds: list[int]


class SplitsConfig(_Frozen):
    """Calibration/test manifests (disjointness enforced at eval time)."""

    calib: SplitManifest
    test: SplitManifest


class EvalConfig(_Frozen):
    """The condition matrix to evaluate plus the calibration/test splits."""

    matrix: list[str] = Field(min_length=1)
    splits: SplitsConfig


class Config(_Frozen):
    """A fully-pinned run configuration (the snapshot logged with every run)."""

    seed: int
    model: ModelConfig
    env: EnvConfig
    attack: AttackConfig
    metric: MetricConfig
    detector: DetectorConfig
    eval: EvalConfig


def load_config(path: StrPath) -> Config:
    """Load and validate a pinned-config YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        pydantic.ValidationError: If any field is missing, out of range, or
            unknown (the message names the offending field).
    """
    data = yaml.safe_load(Path(path).read_text())
    return Config.model_validate(data)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into ``{dotted.path: leaf_value}`` (lists are leaves)."""
    out: dict = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def one_variable_diff(cfg_a: Config, cfg_b: Config) -> list[str]:
    """Return the sorted dotted paths of leaf fields that differ between configs.

    Lists are compared as whole values (a changed list counts as one path), so
    the result has one entry per genuinely changed knob — supporting the
    one-variable-at-a-time run discipline.

    Args:
        cfg_a: The baseline config.
        cfg_b: The config to compare against it.

    Returns:
        Sorted dotted leaf paths whose values differ (empty if identical).
    """
    flat_a = _flatten(cfg_a.model_dump())
    flat_b = _flatten(cfg_b.model_dump())
    keys = set(flat_a) | set(flat_b)
    return sorted(k for k in keys if flat_a.get(k) != flat_b.get(k))
