---
source_file: "src/evasion_tax/attack/__init__.py"
type: "code"
community: "Idealized action-space attacker package "
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Idealized_action-space_attacker_package_
---

# __init__.py

## Connections
- [[Idealized action-space attacker package (playbook §4b-(II), mechanism M-b).  The]] - `rationale_for` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Idealized_action-space_attacker_package_

## 📄 Source

`src/evasion_tax/attack/__init__.py`

```python
"""Idealized action-space attacker package (playbook §4b-(II), mechanism M-b).

The model-free intrinsic-tax instrument: a direct action-space search that
maximises target-reach while minimising the metric-A oracle consistency score,
tracing the ``(ASR, evasion)`` Pareto frontier (hypothesis H6-A, the M3 floor).
The optimiser + frontier logic are model-free (local, synthetic dynamics);
reachability / privileged-state on real scenes is the deferred GPU/LIBERO part
behind the :class:`~evasion_tax.attack.dynamics.Dynamics` seam.
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
    "IdealizedActionAttacker",
    "RealDynamics",
    "Scorer",
    "SyntheticDynamics",
    "asr_at_evasion",
    "pareto_frontier",
    "trace_frontier",
]
```

