---
type: community
members: 32
---

# Goal-Agnostic Anomaly Baseline

**Members:** 32 nodes

## Members
- [[.__post_init__()_5]] - code - src/evasion_tax/baselines/anomaly.py
- [[.as_arrays()]] - code - src/evasion_tax/baselines/anomaly.py
- [[.from_rollouts()]] - code - src/evasion_tax/baselines/anomaly.py
- [[A benign rollout small zero-mean action deltas (typical LIBERO motion).]] - rationale - tests/evasion_tax/baselines/test_anomaly.py
- [[A minimal RolloutStep carrying just what the anomaly baseline reads.      The an]] - rationale - tests/evasion_tax/baselines/test_anomaly.py
- [[BenignActionStats]] - code - src/evasion_tax/baselines/anomaly.py
- [[Estimate per-dim meanstd from the pooled actions of benign rollouts.          A]] - rationale - src/evasion_tax/baselines/anomaly.py
- [[Fair-comparison via shared calibrate (plan invariant 4)]] - rationale - src/evasion_tax/detector/calibrate.py
- [[Goal-agnostic action-anomaly baseline (Task 8).  The mandatory M2 comparison]] - rationale - src/evasion_tax/baselines/anomaly.py
- [[One causal, goal-agnostic anomaly score per step (higher = more anomalous).]] - rationale - src/evasion_tax/baselines/anomaly.py
- [[Per-dimension benign action statistics (immutable; plan invariant 6).      Stor]] - rationale - src/evasion_tax/baselines/anomaly.py
- [[Return ``(mean, std)`` as float ndarrays.]] - rationale - src/evasion_tax/baselines/anomaly.py
- [[Tests for the goal-agnostic action-anomaly baseline (Task 8).  This is the man]] - rationale - tests/evasion_tax/baselines/test_anomaly.py
- [[anomaly.py]] - code - src/evasion_tax/baselines/anomaly.py
- [[benign_rollout()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[benign_stats_from()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[goal_agnostic_anomaly_score()]] - code - src/evasion_tax/baselines/anomaly.py
- [[make_step()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[rollout_from_actions()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_action_at_benign_mean_scores_low()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_anomaly.py]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_benign_stats_from_rollouts_has_per_dim_mean_and_std()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_benign_stats_is_immutable()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_benign_stats_rejects_empty_rollouts()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_calibrates_through_shared_calibrate_and_flags_attack()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_degenerate_benign_with_no_variation_abstains_with_zero_scores()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_does_not_mutate_rollout_or_stats()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_empty_rollout_returns_no_scores()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_out_of_distribution_actions_score_higher_than_benign()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_returns_one_score_per_step_in_unit_interval()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_score_at_t_is_unaffected_by_future_steps()]] - code - tests/evasion_tax/baselines/test_anomaly.py
- [[test_score_is_independent_of_trusted_goal_and_target_region()]] - code - tests/evasion_tax/baselines/test_anomaly.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Goal-Agnostic_Anomaly_Baseline
SORT file.name ASC
```

## Connections to other communities
- 13 edges to [[_COMMUNITY_Rollout]]
- 3 edges to [[_COMMUNITY_Detector Decision Logic]]
- 3 edges to [[_COMMUNITY_Detector Calibration]]
- 2 edges to [[_COMMUNITY_Oracle Frontier Tests]]
- 1 edge to [[_COMMUNITY_L0 Perplexity Filter]]

## Top bridge nodes
- [[goal_agnostic_anomaly_score()]] - degree 21, connects to 4 communities
- [[test_anomaly.py]] - degree 24, connects to 3 communities
- [[BenignActionStats]] - degree 10, connects to 2 communities
- [[test_calibrates_through_shared_calibrate_and_flags_attack()]] - degree 6, connects to 2 communities
- [[Fair-comparison via shared calibrate (plan invariant 4)]] - degree 3, connects to 2 communities