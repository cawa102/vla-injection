"""Attack package — the model-free intrinsic-tax instrument + the GCG search core.

Two model-free attackers share this package, both behind a thin model seam so the
algorithm stays torch-free and toy-testable:

* the action-space **idealized** attacker (playbook §4b-(II), mechanism M-b): a
  direct action-space search that maximises target-reach while minimising the
  metric-A oracle consistency score, tracing the ``(ASR, evasion)`` Pareto frontier
  (hypothesis H6-A, the M3 floor), behind the
  :class:`~evasion_tax.attack.dynamics.Dynamics` seam;
* the **GCG** suffix-search core (step-6 Task 1): the one-hot top-k +
  candidate-batch loop behind the :class:`~evasion_tax.attack.gcg.LossGradientFn`
  seam, whose GPU implementation is ``evasion_tax.attack.gcg_openvla``.
"""

from evasion_tax.attack.dynamics import (
    AttackScenario,
    Dynamics,
    RealDynamics,
    SyntheticDynamics,
)
from evasion_tax.attack.frontier import (
    Frontier,
    FrontierPoint,
    asr_at_evasion,
    pareto_frontier,
)
from evasion_tax.attack.gcg import (
    GcgConfig,
    GcgResult,
    LossGradientFn,
    run_gcg,
    sample_candidates,
    select_best,
    top_k_candidates,
)
from evasion_tax.attack.idealized_frontier import (
    AttackResult,
    IdealizedActionAttacker,
    Scorer,
    trace_frontier,
)

__all__ = [
    "AttackResult",
    "AttackScenario",
    "Dynamics",
    "Frontier",
    "FrontierPoint",
    "GcgConfig",
    "GcgResult",
    "IdealizedActionAttacker",
    "LossGradientFn",
    "RealDynamics",
    "Scorer",
    "SyntheticDynamics",
    "asr_at_evasion",
    "pareto_frontier",
    "run_gcg",
    "sample_candidates",
    "select_best",
    "top_k_candidates",
    "trace_frontier",
]
