---
type: community
cohesion: 0.33
members: 6
---

# Metric (A) Schema Tests

**Cohesion:** 0.33 - loosely connected
**Members:** 6 nodes

## Members
- [[Frozen metric-(A) annotation schema (v1). No attack-tuned values.      Attribute]] - rationale - src/evasion_tax/metric/consistency_a.py
- [[SchemaA]] - code - src/evasion_tax/metric/consistency_a.py
- [[test_metric_rejects_bad_k()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_schema_and_semantics_and_anchor_are_immutable()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_schema_defaults_match_frozen_doc()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_schema_rejects_unknown_combination()]] - code - tests/evasion_tax/metric/test_consistency_a.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Metric_A_Schema_Tests
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Metric (A) Tests]]
- 5 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 4 edges to [[_COMMUNITY_Metric Separation Demo]]
- 3 edges to [[_COMMUNITY_Core Records & Protocols]]
- 3 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]
- 1 edge to [[_COMMUNITY_Oracle Frontier Tests]]
- 1 edge to [[_COMMUNITY_Attack Dynamics & Rollout]]
- 1 edge to [[_COMMUNITY_Privileged State Adapter]]
- 1 edge to [[_COMMUNITY_Synthetic State Fixtures]]

## Top bridge nodes
- [[SchemaA]] - degree 25, connects to 9 communities
- [[test_schema_and_semantics_and_anchor_are_immutable()]] - degree 4, connects to 2 communities
- [[test_metric_rejects_bad_k()]] - degree 3, connects to 2 communities
- [[test_schema_rejects_unknown_combination()]] - degree 3, connects to 2 communities
- [[test_schema_defaults_match_frozen_doc()]] - degree 2, connects to 1 community