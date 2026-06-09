---
type: community
members: 26
---

# Eval Harness & Power

**Members:** 26 nodes

## Members
- [[A minimal OperatingPoint carrying only the fields the rule reads.]] - rationale - tests/evasion_tax/eval/test_power.py
- [[Attach a class`PowerStatus` to each operating point, in order.      The report]] - rationale - src/evasion_tax/eval/power.py
- [[Classify one operating point as powered  primary against the rule.      Args]] - rationale - src/evasion_tax/eval/power.py
- [[Minimum held-out benign N to support a per-rollout FPR claim of ``fpr_target``.]] - rationale - src/evasion_tax/eval/power.py
- [[Operating-point power  sample-size rule (Codex review 2 3).  The detector rep]] - rationale - src/evasion_tax/eval/power.py
- [[OperatingPoint_1]] - code - src/evasion_tax/eval/power.py
- [[RULE_OF_THREE_EVENTS]] - code - src/evasion_tax/eval/power.py
- [[Tests for the operating-point power  sample-size rule (Codex review 2 3).  Th]] - rationale - tests/evasion_tax/eval/test_power.py
- [[_op()]] - code - tests/evasion_tax/eval/test_power.py
- [[annotate_operating_points()]] - code - src/evasion_tax/eval/power.py
- [[classify_power()]] - code - src/evasion_tax/eval/power.py
- [[power.py]] - code - src/evasion_tax/eval/power.py
- [[required_benign_n()]] - code - src/evasion_tax/eval/power.py
- [[test_annotate_operating_points_classifies_each_point()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_annotate_uses_held_out_n_benign_not_calib()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_power.py]] - code - tests/evasion_tax/eval/test_power.py
- [[test_power_status_is_immutable()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_powered_boundary_is_inclusive_at_required_n()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_primary_flag_tracks_primary_fpr()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_primary_point_is_powered_at_modest_n()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_required_benign_n_honours_min_events_override()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_required_benign_n_matches_playbook_floors()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_required_benign_n_rejects_out_of_range_fpr()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_required_benign_n_rounds_up()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_rule_of_three_constant_is_three()]] - code - tests/evasion_tax/eval/test_power.py
- [[test_underpowered_tight_point_is_flagged_not_silent()]] - code - tests/evasion_tax/eval/test_power.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Eval_Harness__Power
SORT file.name ASC
```

## Connections to other communities
- 9 edges to [[_COMMUNITY_Eval Harness & Power]]
- 5 edges to [[_COMMUNITY_Eval Harness & Power]]
- 4 edges to [[_COMMUNITY_run_condition_matrix]]
- 1 edge to [[_COMMUNITY_Eval Harness & Power]]
- 1 edge to [[_COMMUNITY_CalibTest Split Disjointness]]

## Top bridge nodes
- [[test_power.py]] - degree 27, connects to 2 communities
- [[power.py]] - degree 15, connects to 2 communities
- [[annotate_operating_points()]] - degree 9, connects to 2 communities
- [[classify_power()]] - degree 11, connects to 1 community
- [[required_benign_n()]] - degree 9, connects to 1 community