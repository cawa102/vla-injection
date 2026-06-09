---
type: community
members: 66
---

# Detector Metrics & CIs

**Members:** 66 nodes

## Members
- [[A binomial proportion confidence interval for ``k`` of ``n``.      Args]] - rationale - src/evasion_tax/eval/metrics.py
- [[A calib set with low benign scores and a disjoint held-out (eval) set     whos]] - rationale - tests/evasion_tax/eval/test_metrics.py
- [[Benign rollouts (low max-score) vs attacked rollouts (high max-score).      Each]] - rationale - tests/evasion_tax/eval/test_metrics.py
- [[Drop in benign task-success caused by enabling the detector.]] - rationale - src/evasion_tax/eval/metrics.py
- [[Evaluation statistics for the floor result (Task 7).  Pure NumPySciPysklearn s]] - rationale - src/evasion_tax/eval/metrics.py
- [[Exact (Clopper-Pearson) interval via the Beta quantile function.      lower = Be]] - rationale - src/evasion_tax/eval/metrics.py
- [[Fraction of rollouts aborted by the detector.      Raises         ValueError I]] - rationale - src/evasion_tax/eval/metrics.py
- [[Map each rollout to its per-rollout (max) score.]] - rationale - src/evasion_tax/eval/metrics.py
- [[ROC curve and AUC for per-rollout scores (benign=0, attacked=1).      Args]] - rationale - src/evasion_tax/eval/metrics.py
- [[Reduce a rollout to one score max over its per-step values.      Accepts either]] - rationale - src/evasion_tax/eval/metrics.py
- [[ScoreLike]] - code - src/evasion_tax/eval/metrics.py
- [[Summarise detection latencies, treating ``None`` as never fired.      Args]] - rationale - src/evasion_tax/eval/metrics.py
- [[TPR (with CI) at each target FPR, with tau chosen by ``calibrate``.      For eac]] - rationale - src/evasion_tax/eval/metrics.py
- [[Tests for eval statistics (Task 7) CIs, ROCAUC, TPR@FPR, summaries.  These are]] - rationale - tests/evasion_tax/eval/test_metrics.py
- [[Wilson score interval for ``k`` successes in ``n`` trials.      center = (phat +]] - rationale - src/evasion_tax/eval/metrics.py
- [[_calib_low_eval_high()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[_clopper_pearson_ci()]] - code - src/evasion_tax/eval/metrics.py
- [[_per_rollout_score()]] - code - src/evasion_tax/eval/metrics.py
- [[_per_rollout_scores()]] - code - src/evasion_tax/eval/metrics.py
- [[_separated_per_rollout()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[_wilson_ci()]] - code - src/evasion_tax/eval/metrics.py
- [[abort_rate()]] - code - src/evasion_tax/eval/metrics.py
- [[benign_degradation()]] - code - src/evasion_tax/eval/metrics.py
- [[detection_latency_summary]] - code - src/evasion_tax/eval/metrics.py
- [[detection_latency_summary()]] - code - src/evasion_tax/eval/metrics.py
- [[metrics.py]] - code - src/evasion_tax/eval/metrics.py
- [[ndarray_5]] - code - src/evasion_tax/eval/metrics.py
- [[proportion_ci()]] - code - src/evasion_tax/eval/metrics.py
- [[roc_auc()]] - code - src/evasion_tax/eval/metrics.py
- [[test_abort_rate_basic()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_abort_rate_rejects_count_above_total()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_abort_rate_rejects_negative_count()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_abort_rate_rejects_zero_total()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_auc_is_about_half_for_identical_distributions()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_auc_is_one_for_perfectly_separated_scores()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_benign_degradation_is_drop_in_success()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_benign_degradation_zero_when_unchanged()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_both_methods_contain_point_estimate()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_calib_fpr_diagnostic_stays_conservative_below_target()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_ci_width_shrinks_as_n_grows()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_ci_within_unit_interval()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_clopper_pearson_matches_known_reference_k8_n10()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_clopper_pearson_wider_than_wilson_at_small_n()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_detection_latency_summary_all_none()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_detection_latency_summary_filters_none()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_heldout_fpr_ci_uses_eval_n()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_k_equals_n_upper_bound_is_one()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_k_zero_lower_bound_is_zero()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_metrics.py]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_operating_point_cis_within_unit_interval()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_operating_point_records_split_sizes()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_proportion_ci_rejects_k_out_of_range()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_proportion_ci_rejects_nonpositive_n()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_proportion_ci_rejects_unknown_method()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_realised_benign_fpr_is_conservative_below_target()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_realised_fpr_falls_back_to_calib_when_no_eval_given()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_realised_fpr_is_measured_on_heldout_benign_eval()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_roc_curve_arrays_are_monotone_nondecreasing()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tau_unchanged_by_benign_eval()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tau_used_equals_calibrate_result()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tpr_at_fpr_accepts_scalar_per_rollout_scores()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tpr_at_fpr_does_not_mutate_input()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tpr_at_fpr_returns_one_operating_point_per_target()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_tpr_high_when_classes_separate()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[test_wilson_matches_known_reference_k8_n10()]] - code - tests/evasion_tax/eval/test_metrics.py
- [[tpr_at_fpr()]] - code - src/evasion_tax/eval/metrics.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Detector_Metrics__CIs
SORT file.name ASC
```

## Connections to other communities
- 8 edges to [[_COMMUNITY_run_condition_matrix]]
- 7 edges to [[_COMMUNITY_Detector Calibration]]
- 4 edges to [[_COMMUNITY_Eval Harness & Power]]
- 3 edges to [[_COMMUNITY_Eval Harness & Power]]
- 3 edges to [[_COMMUNITY_Rollout]]
- 2 edges to [[_COMMUNITY_Metric Separation Demo]]
- 1 edge to [[_COMMUNITY_Figure Generation]]
- 1 edge to [[_COMMUNITY_L1 Internal Probe]]

## Top bridge nodes
- [[tpr_at_fpr()]] - degree 23, connects to 4 communities
- [[roc_auc()]] - degree 11, connects to 4 communities
- [[metrics.py]] - degree 35, connects to 3 communities
- [[test_metrics.py]] - degree 48, connects to 2 communities
- [[ScoreLike]] - degree 6, connects to 2 communities