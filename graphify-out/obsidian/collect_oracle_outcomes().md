---
source_file: "src/evasion_tax/eval/cross_layer.py"
type: "code"
community: "Rollout"
location: "L319"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Rollout
---

# collect_oracle_outcomes()

## Connections
- [[.supported()]] - `calls` [INFERRED]
- [[AttackScenario_2]] - `references` [EXTRACTED]
- [[IdealizedActionAttacker_1]] - `references` [EXTRACTED]
- [[Model-free L2-oracle data path → ``(outcomes, excluded)``.      Runs the §4b-II]] - `rationale_for` [EXTRACTED]
- [[Scorer_1]] - `references` [EXTRACTED]
- [[TargetActionSpec_1]] - `references` [EXTRACTED]
- [[Threshold_2]] - `references` [EXTRACTED]
- [[UnitKey]] - `references` [EXTRACTED]
- [[UnitOutcome]] - `references` [EXTRACTED]
- [[cross_layer.py]] - `contains` [EXTRACTED]
- [[rollout_fires()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_reproduces_trace_frontier()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_sets_blocked_when_detector_fires_in_time()]] - `calls` [INFERRED]
- [[test_collect_oracle_outcomes_surfaces_coverage_excluded()]] - `calls` [INFERRED]
- [[test_cross_layer.py]] - `references` [EXTRACTED]
- [[trace_frontier()]] - `semantically_similar_to` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/Rollout