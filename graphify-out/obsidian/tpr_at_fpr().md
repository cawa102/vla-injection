---
source_file: "src/evasion_tax/eval/metrics.py"
type: "code"
community: "Detector Metrics & CIs"
location: "L193"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Detector_Metrics__CIs
---

# tpr_at_fpr()

## Connections
- [[OperatingPoint]] - `references` [EXTRACTED]
- [[ScoreLike]] - `references` [EXTRACTED]
- [[TPR (with CI) at each target FPR, with tau chosen by ``calibrate``.      For eac]] - `rationale_for` [EXTRACTED]
- [[_per_rollout_scores()]] - `calls` [EXTRACTED]
- [[calibrate()]] - `calls` [INFERRED]
- [[main()_2]] - `calls` [INFERRED]
- [[metrics.py]] - `contains` [EXTRACTED]
- [[proportion_ci()]] - `calls` [EXTRACTED]
- [[run_condition_matrix()]] - `calls` [INFERRED]
- [[test_calib_fpr_diagnostic_stays_conservative_below_target()]] - `calls` [INFERRED]
- [[test_heldout_fpr_ci_uses_eval_n()]] - `calls` [INFERRED]
- [[test_metrics.py]] - `references` [EXTRACTED]
- [[test_operating_point_cis_within_unit_interval()]] - `calls` [INFERRED]
- [[test_operating_point_records_split_sizes()]] - `calls` [INFERRED]
- [[test_realised_benign_fpr_is_conservative_below_target()]] - `calls` [INFERRED]
- [[test_realised_fpr_falls_back_to_calib_when_no_eval_given()]] - `calls` [INFERRED]
- [[test_realised_fpr_is_measured_on_heldout_benign_eval()]] - `calls` [INFERRED]
- [[test_tau_unchanged_by_benign_eval()]] - `calls` [INFERRED]
- [[test_tau_used_equals_calibrate_result()]] - `calls` [INFERRED]
- [[test_tpr_at_fpr_accepts_scalar_per_rollout_scores()]] - `calls` [INFERRED]
- [[test_tpr_at_fpr_does_not_mutate_input()]] - `calls` [INFERRED]
- [[test_tpr_at_fpr_returns_one_operating_point_per_target()]] - `calls` [INFERRED]
- [[test_tpr_high_when_classes_separate()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Detector_Metrics__CIs