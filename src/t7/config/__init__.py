"""Pinned-config schema + runtime guard (Task 9).

The boundary layer for a run's parameters and host capability:

* :class:`Config` and :func:`load_config` — validate a YAML config (frozen,
  unknown-field-rejecting); :func:`one_variable_diff` reports the leaf paths that
  differ between two configs (the one-variable-at-a-time run discipline).
* :func:`cuda_available` / :func:`gb10_required_message` in
  :mod:`t7.config.runtime` — the shared GB10 guard the model-dependent scripts
  use to refuse to silently no-op on the local host.
"""

from __future__ import annotations

from t7.config.runtime import cuda_available, gb10_required_message
from t7.config.schema import (
    AttackConfig,
    Config,
    DetectorConfig,
    EnvConfig,
    EvalConfig,
    MetricConfig,
    ModelConfig,
    SplitManifest,
    SplitsConfig,
    load_config,
    one_variable_diff,
)

__all__ = [
    "AttackConfig",
    "Config",
    "DetectorConfig",
    "EnvConfig",
    "EvalConfig",
    "MetricConfig",
    "ModelConfig",
    "SplitManifest",
    "SplitsConfig",
    "cuda_available",
    "gb10_required_message",
    "load_config",
    "one_variable_diff",
]
