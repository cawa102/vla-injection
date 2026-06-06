"""Embodiment Evasion Tax — goal-action consistency detection for instruction-injected VLA policies.

This package holds the **model-free** components that can be built and unit-tested on a
local dev host without CUDA: reproducibility infrastructure, the privileged-state consistency
metric (A), the FP-calibrated detector, evaluation statistics, the OpenVLA action codec,
and baselines.

Model/GPU-dependent pieces (OpenVLA-7B inference, GCG optimisation, LIBERO *rollouts*) are
deferred to the GPU node (A100/H100) and sit behind thin interfaces with synthetic fixtures
for tests.

See ``docs/core/local-prep-plan.md`` for the task breakdown.
"""

__version__ = "0.0.0"
