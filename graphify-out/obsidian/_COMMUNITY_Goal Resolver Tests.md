---
type: community
cohesion: 0.28
members: 9
---

# Goal Resolver Tests

**Cohesion:** 0.28 - loosely connected
**Members:** 9 nodes

## Members
- [[A throwaway step; ``PrivilegedGoalResolver`` is step-agnostic (anchors on state)]] - rationale - tests/evasion_tax/metric/test_state_libero.py
- [[PrivilegedGoalResolver]] - code - src/evasion_tax/metric/consistency_a.py
- [[Resolve the anchor from sim ground truth (privileged → non-deployable).      ``a]] - rationale - src/evasion_tax/metric/consistency_a.py
- [[RolloutStep_2]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[_dummy_step()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_resolver_abstains_on_region_without_pose()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_resolver_resolves_plate_anchor()]] - code - tests/evasion_tax/metric/test_state_libero.py
- [[test_resolver_returns_anchor_at_target_region_object()]] - code - tests/evasion_tax/metric/test_consistency_a.py
- [[test_resolver_unresolvable_returns_none()]] - code - tests/evasion_tax/metric/test_consistency_a.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Goal_Resolver_Tests
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Metric (A) Tests]]
- 4 edges to [[_COMMUNITY_LIBERO State Adapter Tests]]
- 3 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 2 edges to [[_COMMUNITY_Core Records & Protocols]]
- 2 edges to [[_COMMUNITY_Attack Dynamics & Rollout]]
- 2 edges to [[_COMMUNITY_Privileged State Adapter]]
- 1 edge to [[_COMMUNITY_Synthetic State Fixtures]]

## Top bridge nodes
- [[PrivilegedGoalResolver]] - degree 16, connects to 7 communities
- [[RolloutStep_2]] - degree 4, connects to 2 communities
- [[_dummy_step()]] - degree 5, connects to 1 community
- [[test_resolver_returns_anchor_at_target_region_object()]] - degree 3, connects to 1 community
- [[test_resolver_unresolvable_returns_none()]] - degree 3, connects to 1 community