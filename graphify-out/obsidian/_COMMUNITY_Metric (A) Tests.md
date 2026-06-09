---
type: community
members: 33
---

# Metric (A) Tests

**Members:** 33 nodes

## Members
- [[A privileged_state dict in the normalised PrivilegedState schema.]] - rationale - tests/evasion_tax/metric/test_consistency_a.py
- [[Tests for consistency metric (A) — frozen schema + causal scorer (Task 5).  Thes]] - rationale - tests/evasion_tax/metric/test_consistency_a.py
- [[_anchor()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[_metric()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[_priv()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[_rollout()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[_step()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_combination_max_is_default_and_takes_worst_primitive()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_combination_noisy_or_and_mean_match_formulas()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_consistency_a.py]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_extract_semantics_returns_zeroed_on_unresolved_anchor()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_k_controls_how_many_past_steps_inform_the_score()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_monitoring_ceiling_can_see_future_deviation_that_causal_misses()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_monitoring_ceiling_is_a_true_upper_bound_on_causal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_monitoring_ceiling_returns_one_score_per_step()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p1_high_when_moving_away_from_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p1_low_when_moving_toward_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p1_zero_when_at_goal_regardless_of_direction()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p1_zero_when_no_motion_and_not_at_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p2_high_when_engaging_distractor_far_from_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p2_zero_when_near_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p2_zero_when_no_distractors()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p3_high_for_grasp_at_distractor()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p3_low_for_grasp_at_goal()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_p3_zero_without_a_grasp_event()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_resolver_unresolvable_returns_none()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_does_not_warn_when_anchor_resolves()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_is_causal_future_steps_do_not_change_score_at_t()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_logs_warning_on_runtime_abstain()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_returns_valid_score_with_window_end_equal_step_index()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_rollout_has_one_score_per_step_in_order()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_score_uses_exactly_the_causal_window_steps()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_unresolvable_goal_scores_zero()]] - code - tests/evasion_tax/metric/test_consistency_a.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Metric_A_Tests
SORT file.name ASC
```

## Connections to other communities
- 10 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 7 edges to [[_COMMUNITY_Metric Separation Demo]]
- 5 edges to [[_COMMUNITY_Privileged State Adapter]]
- 1 edge to [[_COMMUNITY_Rollout]]
- 1 edge to [[_COMMUNITY_LIBERO State Adapter Tests]]
- 1 edge to [[_COMMUNITY_Oracle Frontier Tests]]
- 1 edge to [[_COMMUNITY_L1 Internal Probe]]
- 1 edge to [[_COMMUNITY_Detector Decision Logic]]

## Top bridge nodes
- [[test_consistency_a.py]] - degree 48, connects to 8 communities
- [[_metric()]] - degree 26, connects to 2 communities
- [[test_combination_noisy_or_and_mean_match_formulas()]] - degree 7, connects to 2 communities
- [[test_k_controls_how_many_past_steps_inform_the_score()]] - degree 5, connects to 2 communities
- [[_step()]] - degree 28, connects to 1 community