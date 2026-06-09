---
type: community
cohesion: 0.25
members: 8
---

# Oracle Outcome Collection

**Cohesion:** 0.25 - loosely connected
**Members:** 8 nodes

## Members
- [[CoverageCell_1]] - code - src/evasion_tax/metric/coverage.py
- [[CoverageManifest_1]] - code - src/evasion_tax/metric/coverage.py
- [[CoverageStatus_1]] - code - src/evasion_tax/metric/coverage.py
- [[GoalKind_1]] - code - src/evasion_tax/metric/coverage.py
- [[L0L1L2 cross-layer evasion-tax frontier]] - concept - src/evasion_tax/eval/cross_layer.py
- [[build_manifest]] - code - src/evasion_tax/metric/coverage.py
- [[classify_cell]] - code - src/evasion_tax/metric/coverage.py
- [[collect_oracle_outcomes (L2-oracle data path)]] - code - src/evasion_tax/eval/cross_layer.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Oracle_Outcome_Collection
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Target Action Spec]]
- 1 edge to [[_COMMUNITY_Tax Estimation (delta-ASR)]]
- 1 edge to [[_COMMUNITY_Metric (A) Oracle Design]]
- 1 edge to [[_COMMUNITY_Eval Harness Concepts]]

## Top bridge nodes
- [[collect_oracle_outcomes (L2-oracle data path)]] - degree 6, connects to 4 communities