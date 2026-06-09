---
type: community
cohesion: 0.25
members: 11
---

# Eval Harness Concepts

**Cohesion:** 0.25 - loosely connected
**Members:** 11 nodes

## Members
- [[ConditionRow_1]] - code - src/evasion_tax/eval/harness.py
- [[No-leakage calibtest invariant 3]] - concept - src/evasion_tax/eval/splits.py
- [[OperatingPoint_3]] - code - src/evasion_tax/eval/metrics.py
- [[PowerStatus_1]] - code - src/evasion_tax/eval/power.py
- [[_per_rollout_scores]] - code - src/evasion_tax/eval/metrics.py
- [[annotate_operating_points]] - code - src/evasion_tax/eval/power.py
- [[assert_disjoint (calibtest leakage guard)]] - code - src/evasion_tax/eval/splits.py
- [[classify_power]] - code - src/evasion_tax/eval/power.py
- [[required_benign_n (rule of three)]] - code - src/evasion_tax/eval/power.py
- [[run_condition_matrix]] - code - src/evasion_tax/eval/harness.py
- [[tpr_at_fpr]] - code - src/evasion_tax/eval/metrics.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Eval_Harness_Concepts
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Figure Pipeline Concepts]]
- 1 edge to [[_COMMUNITY_Oracle Outcome Collection]]
- 1 edge to [[_COMMUNITY_Binomial CI Helpers]]
- 1 edge to [[_COMMUNITY_Detector Calibration]]
- 1 edge to [[_COMMUNITY_L1 Probe Instrument]]

## Top bridge nodes
- [[run_condition_matrix]] - degree 8, connects to 2 communities
- [[tpr_at_fpr]] - degree 5, connects to 2 communities
- [[_per_rollout_scores]] - degree 3, connects to 1 community