---
source_file: "src/evasion_tax/detector/decide.py"
type: "code"
community: "Detector Decision Logic"
location: "L34"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Detector_Decision_Logic
---

# rollout_fires()

## Connections
- [[Decision]] - `references` [EXTRACTED]
- [[Decision (shared record)]] - `references` [EXTRACTED]
- [[Return the first-exceedance decision for one ordered rollout.      Scans ``score]] - `rationale_for` [EXTRACTED]
- [[Score_4]] - `references` [EXTRACTED]
- [[collect_oracle_outcomes()]] - `calls` [INFERRED]
- [[decide.py]] - `contains` [EXTRACTED]
- [[per_rollout_fire_rate()]] - `calls` [INFERRED]
- [[test_anomaly.py]] - `references` [EXTRACTED]
- [[test_benign_weird_suffix_does_not_fire_at_calibrated_threshold()]] - `calls` [INFERRED]
- [[test_calibrate.py]] - `references` [EXTRACTED]
- [[test_calibrates_identically_through_shared_calibrate()]] - `calls` [INFERRED]
- [[test_calibrates_through_shared_calibrate_and_flags_attack()]] - `calls` [INFERRED]
- [[test_calibrates_through_shared_calibrate_and_flags_injection()]] - `calls` [INFERRED]
- [[test_decide.py]] - `references` [EXTRACTED]
- [[test_idealized_frontier.py]] - `references` [EXTRACTED]
- [[test_perplexity.py]] - `references` [EXTRACTED]
- [[test_probe_internal.py]] - `references` [EXTRACTED]
- [[test_rollout_fires_does_not_mutate_input()]] - `calls` [INFERRED]
- [[test_rollout_fires_empty_rollout_does_not_fire()]] - `calls` [INFERRED]
- [[test_rollout_fires_is_causal_first_not_max()]] - `calls` [INFERRED]
- [[test_rollout_fires_no_exceedance_returns_hold_false_step_minus_one()]] - `calls` [INFERRED]
- [[test_rollout_fires_returns_first_exceedance_step()]] - `calls` [INFERRED]
- [[test_trace_frontier_benign_fpr_is_conservative()]] - `calls` [INFERRED]
- [[trace_frontier()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Detector_Decision_Logic