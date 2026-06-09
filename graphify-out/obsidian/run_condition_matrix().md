---
source_file: "src/evasion_tax/eval/harness.py"
type: "code"
community: "Eval Harness & Power"
location: "L74"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Eval_Harness__Power
---

# run_condition_matrix()

## Connections
- [[ConditionRow]] - `calls` [EXTRACTED]
- [[ConditionSplits]] - `references` [EXTRACTED]
- [[Evaluate every condition calibrate on calib, score on disjoint test.      Args]] - `rationale_for` [EXTRACTED]
- [[ResultsTable_1]] - `references` [EXTRACTED]
- [[_per_rollout_scores()]] - `calls` [INFERRED]
- [[annotate_operating_points()]] - `calls` [INFERRED]
- [[harness.py]] - `contains` [EXTRACTED]
- [[main()_1]] - `calls` [INFERRED]
- [[main()_4]] - `calls` [INFERRED]
- [[roc_auc()]] - `calls` [INFERRED]
- [[test_each_row_has_operating_points_per_target()]] - `calls` [INFERRED]
- [[test_each_row_has_power_status_aligned_with_operating_points()]] - `calls` [INFERRED]
- [[test_empty_conditions_yields_empty_table()]] - `calls` [INFERRED]
- [[test_figures.py]] - `references` [EXTRACTED]
- [[test_fpr_targets_and_primary_fpr_are_threaded()]] - `calls` [INFERRED]
- [[test_harness.py]] - `references` [EXTRACTED]
- [[test_overlapping_condition_has_auc_near_half()]] - `calls` [INFERRED]
- [[test_realised_fpr_is_measured_on_heldout_benign_test_not_calib()]] - `calls` [INFERRED]
- [[test_returns_results_table_with_one_row_per_condition()]] - `calls` [INFERRED]
- [[test_row_has_deferred_latency_summary()]] - `calls` [INFERRED]
- [[test_run_condition_matrix_does_not_mutate_input()]] - `calls` [INFERRED]
- [[test_separated_condition_has_high_auc()]] - `calls` [INFERRED]
- [[test_serialised_table_carries_power_block_flagging_underpowered_points()]] - `calls` [INFERRED]
- [[test_serialised_table_round_trips_into_make_figures()]] - `calls` [INFERRED]
- [[test_table_retains_raw_arrays_for_figures()]] - `calls` [INFERRED]
- [[test_tau_in_table_comes_from_calibrate_on_benign_calib_only()]] - `calls` [INFERRED]
- [[tpr_at_fpr()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/Eval_Harness__Power