---
type: community
members: 26
---

# LIBERO State Adapter Tests

**Members:** 26 nodes

## Members
- [[A throwaway step; ``PrivilegedGoalResolver`` is step-agnostic (anchors on state)]] - rationale - tests/evasion_tax/metric/test_state_libero.py
- [[Tests for the concrete LIBERO StateAdapter (``state_libero.py``).  Run against F]] - rationale - tests/evasion_tax/metric/test_state_libero.py
- [[_dummy_step()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[_load()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[extract_ee_pos]] - code - src/evasion_tax/metric/state_libero.py
- [[extract_object_poses]] - code - src/evasion_tax/metric/state_libero.py
- [[gripper_open_from_qpos]] - code - src/evasion_tax/metric/state_libero.py
- [[opendrawer()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[spatial0()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[state_libero.py]] - code - src/evasion_tax/metric/state_libero.py
- [[target_region_from_obj_of_interest]] - code - src/evasion_tax/metric/state_libero.py
- [[test_adapter_builds_resolvable_state_spatial0()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_adapter_opendrawer_abstain_case()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_adapter_rejects_non_mapping()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_extract_ee_pos_from_real_obs()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_extract_ee_pos_missing_raises()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_gripper_closed_when_qpos_near_zero()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_gripper_open_true_on_real_open_value()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_gripper_threshold_constant()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_object_poses_excludes_relative_and_robot_keys()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_resolver_abstains_on_region_without_pose()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_resolver_resolves_plate_anchor()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_state_libero.py]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_target_region_empty_is_none()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_target_region_is_last_obj_of_interest_binary()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_target_region_unary_predicate()]] - code - tests/evasion_tax/metric/test_state_libero.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/LIBERO_State_Adapter_Tests
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Privileged State Adapter]]
- 3 edges to [[_COMMUNITY_LIBERO Fixtures & Adapter]]
- 1 edge to [[_COMMUNITY_Rollout]]
- 1 edge to [[_COMMUNITY_LIBERO Obs Fixture (Bowls)]]
- 1 edge to [[_COMMUNITY_Metric (A) Tests]]

## Top bridge nodes
- [[test_state_libero.py]] - degree 31, connects to 5 communities
- [[_dummy_step()]] - degree 5, connects to 1 community
- [[state_libero.py]] - degree 5, connects to 1 community
- [[test_resolver_resolves_plate_anchor()]] - degree 3, connects to 1 community
- [[test_resolver_abstains_on_region_without_pose()]] - degree 3, connects to 1 community