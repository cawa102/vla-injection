---
source_file: "tests/evasion_tax/detector/test_calibrate.py"
type: "code"
community: "Detector Calibration"
location: "L18"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Detector_Calibration
---

# benign_rollouts()

## Connections
- [[A synthetic benign calibration set list of rollouts of per-step floats in 0,1]] - `rationale_for` [EXTRACTED]
- [[test_calibrate.py]] - `contains` [EXTRACTED]
- [[test_calibrate_accepts_score_objects()]] - `calls` [EXTRACTED]
- [[test_calibrate_does_not_mutate_input()]] - `calls` [EXTRACTED]
- [[test_calibrate_rejects_target_above_one()]] - `calls` [EXTRACTED]
- [[test_calibrate_rejects_target_below_zero()]] - `calls` [EXTRACTED]
- [[test_calibrate_rejects_unknown_aggregate()]] - `calls` [EXTRACTED]
- [[test_calibrate_returns_threshold_record()]] - `calls` [EXTRACTED]
- [[test_calibrated_per_rollout_fire_rate_at_or_below_target()]] - `calls` [EXTRACTED]
- [[test_per_rollout_and_per_window_give_different_tau()]] - `calls` [EXTRACTED]
- [[test_smaller_target_gives_higher_tau_and_fewer_fires()]] - `calls` [EXTRACTED]
- [[test_target_one_means_almost_everything_fires()]] - `calls` [EXTRACTED]
- [[test_target_zero_means_nothing_fires_on_calibration_set()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Detector_Calibration