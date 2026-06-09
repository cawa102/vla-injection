---
source_file: "src/evasion_tax/baselines/anomaly.py"
type: "code"
community: "Goal-Agnostic Anomaly Baseline"
location: "L87"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Goal-Agnostic_Anomaly_Baseline
---

# goal_agnostic_anomaly_score()

## Connections
- [[.as_arrays()]] - `calls` [EXTRACTED]
- [[BenignActionStats]] - `references` [EXTRACTED]
- [[Fair-comparison via shared calibrate (plan invariant 4)]] - `rationale_for` [INFERRED]
- [[One causal, goal-agnostic anomaly score per step (higher = more anomalous).]] - `rationale_for` [EXTRACTED]
- [[PerplexityFilter]] - `semantically_similar_to` [INFERRED]
- [[Rollout_4]] - `references` [EXTRACTED]
- [[Rollout (shared record)]] - `references` [EXTRACTED]
- [[Score_2]] - `references` [EXTRACTED]
- [[Score (shared record)]] - `references` [EXTRACTED]
- [[Scorer]] - `semantically_similar_to` [INFERRED]
- [[anomaly.py]] - `contains` [EXTRACTED]
- [[test_action_at_benign_mean_scores_low()]] - `calls` [INFERRED]
- [[test_anomaly.py]] - `references` [EXTRACTED]
- [[test_calibrates_through_shared_calibrate_and_flags_attack()]] - `calls` [INFERRED]
- [[test_degenerate_benign_with_no_variation_abstains_with_zero_scores()]] - `calls` [INFERRED]
- [[test_does_not_mutate_rollout_or_stats()]] - `calls` [INFERRED]
- [[test_empty_rollout_returns_no_scores()]] - `calls` [INFERRED]
- [[test_out_of_distribution_actions_score_higher_than_benign()]] - `calls` [INFERRED]
- [[test_returns_one_score_per_step_in_unit_interval()]] - `calls` [INFERRED]
- [[test_score_at_t_is_unaffected_by_future_steps()]] - `calls` [INFERRED]
- [[test_score_is_independent_of_trusted_goal_and_target_region()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Goal-Agnostic_Anomaly_Baseline