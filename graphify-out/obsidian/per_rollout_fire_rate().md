---
source_file: "tests/evasion_tax/detector/test_calibrate.py"
type: "code"
community: "Detector Calibration"
location: "L24"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Detector_Calibration
---

# per_rollout_fire_rate()

## Connections
- [[Empirical per-rollout fire-rate a rollout fires iff ANY step value  tau.]] - `rationale_for` [EXTRACTED]
- [[rollout_fires()]] - `calls` [INFERRED]
- [[test_calibrate.py]] - `contains` [EXTRACTED]
- [[test_calibrate_works_on_arbitrary_made_up_scores()]] - `calls` [EXTRACTED]
- [[test_calibrated_per_rollout_fire_rate_at_or_below_target()]] - `calls` [EXTRACTED]
- [[test_smaller_target_gives_higher_tau_and_fewer_fires()]] - `calls` [EXTRACTED]
- [[test_target_one_means_almost_everything_fires()]] - `calls` [EXTRACTED]
- [[test_target_zero_means_nothing_fires_on_calibration_set()]] - `calls` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Detector_Calibration