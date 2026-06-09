---
type: community
members: 45
---

# Rollout Records Tests

**Members:** 45 nodes

## Members
- [[Build a RolloutStep with sensible defaults; action defaults to a 7-vector.]] - rationale - tests/evasion_tax/test_records.py
- [[Tests for the core immutable data records (Task 2).  Covers frozen-record immut]] - rationale - tests/evasion_tax/test_records.py
- [[make_rollout()_2]] - code - tests/evasion_tax/test_records.py
- [[make_step()_1]] - code - tests/evasion_tax/test_records.py
- [[test_actions_has_shape_n_by_7()]] - code - tests/evasion_tax/test_records.py
- [[test_actions_values_match_steps()]] - code - tests/evasion_tax/test_records.py
- [[test_decision_fields_and_immutability()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_clamped_at_start_when_t_less_than_k_minus_1()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_k_less_than_one_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_mid_rollout_returns_exactly_k_steps_ending_at_t()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_never_includes_future_steps()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_returns_tuple()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_t_out_of_range_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_prefix_window_t_zero_returns_only_first_step()]] - code - tests/evasion_tax/test_records.py
- [[test_records.py]] - code - tests/evasion_tax/test_records.py
- [[test_rollout_is_immutable()]] - code - tests/evasion_tax/test_records.py
- [[test_rollout_len()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_action_accepts_numpy_array()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_action_length_seven_accepted_and_stored_as_float_tuple()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_action_non_numeric_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_action_too_long_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_action_too_short_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_rolloutstep_is_immutable()]] - code - tests/evasion_tax/test_records.py
- [[test_score_accepts_value_in_range()]] - code - tests/evasion_tax/test_records.py
- [[test_score_is_immutable()]] - code - tests/evasion_tax/test_records.py
- [[test_score_rejects_value_above_one()]] - code - tests/evasion_tax/test_records.py
- [[test_score_rejects_value_below_zero()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_dim_out_of_range_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_is_immutable()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_low_above_high_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_mismatched_low_high_dims_lengths_raise()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_persistence_steps_less_than_one_raises()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_false_outside_region()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_inclusive_bounds()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_true_inside_region()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_accepts_numpy_array()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_persistence_greater_than_n_is_false()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_persistence_override()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_requires_consecutive_hits()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_run_of_exactly_persistence_true()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_scattered_hits_false()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_step_agrees_with_reached_window()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_step_first_completion_after_reset()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_step_none_when_never_sustained()]] - code - tests/evasion_tax/test_records.py
- [[test_target_spec_reached_window_step_returns_completion_index()]] - code - tests/evasion_tax/test_records.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Rollout_Records_Tests
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_Rollout]]
- 2 edges to [[_COMMUNITY_Detector Decision Logic]]
- 1 edge to [[_COMMUNITY_Oracle Frontier Tests]]
- 1 edge to [[_COMMUNITY_L1 Internal Probe]]

## Top bridge nodes
- [[test_records.py]] - degree 52, connects to 4 communities
- [[make_rollout()_2]] - degree 14, connects to 1 community
- [[make_step()_1]] - degree 10, connects to 1 community