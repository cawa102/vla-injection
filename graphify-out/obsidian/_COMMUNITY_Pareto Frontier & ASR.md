---
type: community
members: 32
---

# Pareto Frontier & ASR

**Members:** 32 nodes

## Members
- [[(ASR, evasion) Pareto-frontier geometry for §4b-(II) — pure and model-free.  The]] - rationale - src/evasion_tax/attack/frontier.py
- [[ASR achievable at a fixed ``evasion`` level, linearly interpolated.      Outside]] - rationale - src/evasion_tax/attack/frontier.py
- [[Return the non-dominated set, deduplicated and ordered by evasion ascending.]] - rationale - src/evasion_tax/attack/frontier.py
- [[Tests for the (ASR, evasion) Pareto-frontier geometry (playbook §4b-(II)).  Pure]] - rationale - tests/evasion_tax/attack/test_frontier.py
- [[True iff ``q`` Pareto-dominates ``p`` (= on both axes,  on at least one).]] - rationale - src/evasion_tax/attack/frontier.py
- [[_dominates]] - code - src/evasion_tax/attack/frontier.py
- [[_dominates()]] - code - src/evasion_tax/attack/frontier.py
- [[asr_at_evasion()]] - code - src/evasion_tax/attack/frontier.py
- [[frontier.py]] - code - src/evasion_tax/attack/frontier.py
- [[pareto_frontier()]] - code - src/evasion_tax/attack/frontier.py
- [[pt()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_clamps_above_max_evasion_to_min_asr()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_clamps_below_min_evasion_to_max_asr()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_interpolates_between_points()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_is_monotone_non_increasing()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_on_empty_frontier_raises()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_asr_at_evasion_returns_exact_point()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier.py]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_accepts_canonical_shape()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_accepts_empty_and_single()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_is_immutable()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_point_is_immutable()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_point_rejects_negative_tradeoff()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_point_rejects_out_of_range_asr()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_point_rejects_out_of_range_evasion()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_frontier_rejects_non_monotone_points()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_collapses_exact_duplicates_keeping_min_tradeoff()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_drops_dominated_points()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_keeps_only_the_dominating_point()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_of_empty_is_empty()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_of_single_point_is_that_point()]] - code - tests/evasion_tax/attack/test_frontier.py
- [[test_pareto_returns_points_ordered_by_evasion_ascending()]] - code - tests/evasion_tax/attack/test_frontier.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Pareto_Frontier__ASR
SORT file.name ASC
```

## Connections to other communities
- 17 edges to [[_COMMUNITY_Rollout]]
- 4 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]

## Top bridge nodes
- [[pareto_frontier()]] - degree 14, connects to 2 communities
- [[asr_at_evasion()]] - degree 13, connects to 2 communities
- [[test_frontier.py]] - degree 26, connects to 1 community
- [[pt()]] - degree 17, connects to 1 community
- [[frontier.py]] - degree 14, connects to 1 community