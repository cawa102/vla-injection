---
type: community
members: 55
---

# Detector Calibration

**Members:** 55 nodes

## Members
- [[A synthetic benign calibration set list of rollouts of per-step floats in 0,1]] - rationale - tests/evasion_tax/detector/test_calibrate.py
- [[All per-step scores pooled across rollouts (per-window unit).]] - rationale - src/evasion_tax/detector/calibrate.py
- [[Benign +x-ish (goal-directed) rollouts, slightly noisy → small scores.]] - rationale - tests/evasion_tax/attack/test_idealized_frontier.py
- [[Calibrate ``tau`` to a benign false-abort budget.      Args         benign_scor]] - rationale - src/evasion_tax/detector/calibrate.py
- [[Empirical per-rollout fire-rate a rollout fires iff ANY step value  tau.]] - rationale - tests/evasion_tax/detector/test_calibrate.py
- [[Extract a float from a raw score value or a class`Score`.]] - rationale - src/evasion_tax/records.py
- [[One value per rollout its maximum per-step score.]] - rationale - src/evasion_tax/detector/calibrate.py
- [[Reach-the-cube scene cube (goal anchor) at +x, distractor at -x.]] - rationale - tests/evasion_tax/attack/test_idealized_frontier.py
- [[RolloutScores]] - code - src/evasion_tax/detector/calibrate.py
- [[ScoreLike_1]] - code - src/evasion_tax/records.py
- [[Target action region hold a -x position delta for the whole window.]] - rationale - tests/evasion_tax/attack/test_idealized_frontier.py
- [[Tests for FP-calibration of the detector threshold (Task 6).  Covers invariants]] - rationale - tests/evasion_tax/detector/test_calibrate.py
- [[Tests for the idealized action-space attacker + frontier trace (§4b-(II)).  Mech]] - rationale - tests/evasion_tax/attack/test_idealized_frontier.py
- [[_pooled_steps]] - code - src/evasion_tax/detector/calibrate.py
- [[_pooled_steps()]] - code - src/evasion_tax/detector/calibrate.py
- [[_result()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[_rollout_maxima]] - code - src/evasion_tax/detector/calibrate.py
- [[_rollout_maxima()]] - code - src/evasion_tax/detector/calibrate.py
- [[benign_calib()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[benign_rollouts()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[calibrate()]] - code - src/evasion_tax/detector/calibrate.py
- [[make_attacker()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[make_scenario()_1]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[make_scorer()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[make_target()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[ndarray_4]] - code - src/evasion_tax/detector/calibrate.py
- [[per_rollout_fire_rate()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[score_value()]] - code - src/evasion_tax/records.py
- [[test_attack_is_deterministic()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_attack_result_is_immutable()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_attack_result_rejects_negative_tradeoff()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_attack_result_rejects_out_of_range_consistency()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_calibrate.py]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_accepts_score_objects()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_does_not_mutate_input()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_rejects_empty_calibration_set()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_rejects_target_above_one()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_rejects_target_below_zero()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_rejects_unknown_aggregate()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_returns_threshold_record()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrate_works_on_arbitrary_made_up_scores()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_calibrated_per_rollout_fire_rate_at_or_below_target()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_consistency_metric_a_is_a_scorer()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_higher_tradeoff_lowers_selected_consistency()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_idealized_frontier.py]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_per_rollout_and_per_window_give_different_tau()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_reach_greedy_attacker_reaches_target_with_high_consistency()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_reported_reached_is_strict_reached_window()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_smaller_target_gives_higher_tau_and_fewer_fires()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_target_one_means_almost_everything_fires()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_target_zero_means_nothing_fires_on_calibration_set()]] - code - tests/evasion_tax/detector/test_calibrate.py
- [[test_trace_frontier_benign_fpr_is_conservative()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_trace_frontier_excludes_unsupported_scenarios()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_trace_frontier_traces_a_downward_sloping_oracle_frontier()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py
- [[test_trace_frontier_with_all_excluded_returns_empty_frontier()]] - code - tests/evasion_tax/attack/test_idealized_frontier.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Detector_Calibration
SORT file.name ASC
```

## Connections to other communities
- 30 edges to [[_COMMUNITY_Rollout]]
- 7 edges to [[_COMMUNITY_Detector Metrics & CIs]]
- 5 edges to [[_COMMUNITY_Oracle Frontier Tests]]
- 4 edges to [[_COMMUNITY_Detector Decision Logic]]
- 3 edges to [[_COMMUNITY_Goal-Agnostic Anomaly Baseline]]
- 3 edges to [[_COMMUNITY_Cross-Layer Tax Eval]]
- 3 edges to [[_COMMUNITY_L1 Internal Probe]]
- 2 edges to [[_COMMUNITY_Metric Separation Demo]]
- 2 edges to [[_COMMUNITY_L0 Perplexity Filter]]
- 2 edges to [[_COMMUNITY_Eval Harness & Power]]
- 2 edges to [[_COMMUNITY_Metric (A) Consistency Scorer]]
- 1 edge to [[_COMMUNITY_Config Schema & Immutability]]
- 1 edge to [[_COMMUNITY_Eval Harness & Power]]
- 1 edge to [[_COMMUNITY_run_condition_matrix]]

## Top bridge nodes
- [[calibrate()]] - degree 46, connects to 8 communities
- [[score_value()]] - degree 8, connects to 4 communities
- [[test_idealized_frontier.py]] - degree 31, connects to 3 communities
- [[ScoreLike_1]] - degree 8, connects to 3 communities
- [[test_calibrate.py]] - degree 22, connects to 2 communities