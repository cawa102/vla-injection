---
type: community
members: 19
---

# Metric Separation Demo

**Members:** 19 nodes

## Members
- [[A tiny text histogram of scores over 0, 1 for the printed report.]] - rationale - scripts/demo_metric_separation.py
- [[Any]] - code - scripts/demo_metric_separation.py
- [[Frozen metric-(A) annotation schema (v1). No attack-tuned values.      Attribute]] - rationale - src/evasion_tax/metric/consistency_a.py
- [[Generate ``n`` benign + ``n`` attacked rollouts and score each with metric (A).]] - rationale - scripts/demo_metric_separation.py
- [[Generate ``n`` benign + ``n`` attacked rollouts via the chosen backend.      Rol]] - rationale - scripts/demo_metric_separation.py
- [[Per-rollout score = max per-step inconsistency (matches the eval convention).]] - rationale - scripts/demo_metric_separation.py
- [[Rollout]] - code - scripts/demo_metric_separation.py
- [[SchemaA]] - code - src/evasion_tax/metric/consistency_a.py
- [[Score]] - code - scripts/demo_metric_separation.py
- [[_generate()]] - code - scripts/demo_metric_separation.py
- [[_histogram()]] - code - scripts/demo_metric_separation.py
- [[_max_score()]] - code - scripts/demo_metric_separation.py
- [[_print_report()]] - code - scripts/demo_metric_separation.py
- [[demo_metric_separation.py]] - code - scripts/demo_metric_separation.py
- [[generate_scored()]] - code - scripts/demo_metric_separation.py
- [[main()_2]] - code - scripts/demo_metric_separation.py
- [[test_metric_rejects_bad_k()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_schema_defaults_match_frozen_doc()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_schema_rejects_unknown_combination()]] - code - tests/evasion_tax/metric/test_consistency_a.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Metric_Separation_Demo
SORT file.name ASC
```

## Connections to other communities
- 10 edges to [[_COMMUNITY_Rollout]]
- 10 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 7 edges to [[_COMMUNITY_Metric (A) Tests]]
- 6 edges to [[_COMMUNITY_Run Logging & Rollout Demo]]
- 3 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]
- 2 edges to [[_COMMUNITY_Detector Calibration]]
- 2 edges to [[_COMMUNITY_Detector Metrics & CIs]]
- 2 edges to [[_COMMUNITY_Config Schema & Immutability]]
- 1 edge to [[_COMMUNITY_Figure Generation]]
- 1 edge to [[_COMMUNITY_CalibTest Split Disjointness]]
- 1 edge to [[_COMMUNITY_Privileged State Adapter]]
- 1 edge to [[_COMMUNITY_Synthetic State Fixtures]]

## Top bridge nodes
- [[SchemaA]] - degree 25, connects to 7 communities
- [[main()_2]] - degree 11, connects to 4 communities
- [[generate_scored()]] - degree 9, connects to 3 communities
- [[Score]] - degree 8, connects to 3 communities
- [[Rollout]] - degree 7, connects to 3 communities