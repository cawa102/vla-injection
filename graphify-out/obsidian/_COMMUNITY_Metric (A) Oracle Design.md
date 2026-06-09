---
type: community
cohesion: 0.25
members: 9
---

# Metric (A) Oracle Design

**Cohesion:** 0.25 - loosely connected
**Members:** 9 nodes

## Members
- [[ConsistencyMetricA (L2 oracle scorer)]] - code - src/evasion_tax/metric/consistency_a.py
- [[GoalAnchor_1]] - code - src/evasion_tax/metric/consistency_a.py
- [[GoalResolver (Protocol seam)]] - code - src/evasion_tax/metric/consistency_a.py
- [[PrivilegedGoalResolver_1]] - code - src/evasion_tax/metric/consistency_a.py
- [[PrivilegedState (env-agnostic snapshot)]] - code - src/evasion_tax/metric/state.py
- [[SchemaA (frozen metric-A schema)]] - code - src/evasion_tax/metric/consistency_a.py
- [[Semantics (3 inconsistency primitives)]] - code - src/evasion_tax/metric/consistency_a.py
- [[StateAdapter (Protocol seam)]] - code - src/evasion_tax/metric/state.py
- [[SyntheticStateAdapter_1]] - code - src/evasion_tax/metric/state.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Metric_A_Oracle_Design
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Core Records & Protocols]]
- 1 edge to [[_COMMUNITY_Oracle Outcome Collection]]
- 1 edge to [[_COMMUNITY_L1 Probe Instrument]]

## Top bridge nodes
- [[ConsistencyMetricA (L2 oracle scorer)]] - degree 7, connects to 2 communities
- [[SyntheticStateAdapter_1]] - degree 3, connects to 1 community