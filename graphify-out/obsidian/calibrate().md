---
source_file: "src/evasion_tax/detector/calibrate.py"
type: "code"
community: "Detector Calibration"
location: "L67"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Detector_Calibration
---

# calibrate()

## Connections
- [[Calibrate ``tau`` to a benign false-abort budget.      Args         benign_scor]] - `rationale_for` [EXTRACTED]
- [[Fair-comparison via shared calibrate (plan invariant 4)]] - `rationale_for` [INFERRED]
- [[RolloutScores]] - `references` [EXTRACTED]
- [[Threshold_1]] - `references` [EXTRACTED]
- [[_pooled_steps]] - `calls` [EXTRACTED]
- [[_pooled_steps()]] - `calls` [EXTRACTED]
- [[_rollout_maxima]] - `calls` [EXTRACTED]
- [[_rollout_maxima()]] - `calls` [EXTRACTED]
- [[calibrate.py_1]] - `contains` [EXTRACTED]
- [[main()]] - `calls` [INFERRED]
- [[test_anomaly.py]] - `references` [EXTRACTED]
- [[test_benign_weird_suffix_does_not_fire_at_calibrated_threshold()]] - `calls` [INFERRED]
- [[test_calibrate.py]] - `references` [EXTRACTED]
- [[test_calibrate_accepts_score_objects()]] - `calls` [INFERRED]
- [[test_calibrate_does_not_mutate_input()]] - `calls` [INFERRED]
- [[test_calibrate_rejects_empty_calibration_set()]] - `calls` [INFERRED]
- [[test_calibrate_rejects_target_above_one()]] - `calls` [INFERRED]
- [[test_calibrate_rejects_target_below_zero()]] - `calls` [INFERRED]
- [[test_calibrate_rejects_unknown_aggregate()]] - `calls` [INFERRED]
- [[test_calibrate_returns_threshold_record()]] - `calls` [INFERRED]
- [[test_calibrate_works_on_arbitrary_made_up_scores()]] - `calls` [INFERRED]
- [[test_calibrated_per_rollout_fire_rate_at_or_below_target()]] - `calls` [INFERRED]
- [[test_calibrates_identically_through_shared_calibrate()]] - `calls` [INFERRED]
- [[test_calibrates_through_shared_calibrate_and_flags_attack()]] - `calls` [INFERRED]
- [[test_calibrates_through_shared_calibrate_and_flags_injection()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_reproduces_trace_frontier()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_sets_blocked_when_detector_fires_in_time()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_surfaces_coverage_excluded()]] - `calls` [INFERRED]
- [[test_harness.py]] - `references` [EXTRACTED]
- [[test_idealized_frontier.py]] - `references` [EXTRACTED]
- [[test_metrics.py]] - `references` [EXTRACTED]
- [[test_per_rollout_and_per_window_give_different_tau()]] - `calls` [INFERRED]
- [[test_perplexity.py]] - `references` [EXTRACTED]
- [[test_probe_internal.py]] - `references` [EXTRACTED]
- [[test_smaller_target_gives_higher_tau_and_fewer_fires()]] - `calls` [INFERRED]
- [[test_target_one_means_almost_everything_fires()]] - `calls` [INFERRED]
- [[test_target_zero_means_nothing_fires_on_calibration_set()]] - `calls` [INFERRED]
- [[test_tau_in_table_comes_from_calibrate_on_benign_calib_only()]] - `calls` [INFERRED]
- [[test_tau_unchanged_by_benign_eval()]] - `calls` [INFERRED]
- [[test_tau_used_equals_calibrate_result()]] - `calls` [INFERRED]
- [[test_trace_frontier_benign_fpr_is_conservative()]] - `calls` [INFERRED]
- [[test_trace_frontier_excludes_unsupported_scenarios()]] - `calls` [INFERRED]
- [[test_trace_frontier_traces_a_downward_sloping_oracle_frontier()]] - `calls` [INFERRED]
- [[test_trace_frontier_with_all_excluded_returns_empty_frontier()]] - `calls` [INFERRED]
- [[tpr_at_fpr()]] - `calls` [INFERRED]
- [[trace_frontier()]] - `conceptually_related_to` [INFERRED]

#graphify/code #graphify/INFERRED #community/Detector_Calibration