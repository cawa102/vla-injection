---
source_file: "src/evasion_tax/baselines/perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L86"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# PerplexityFilter

## Connections
- [[.__init__()_1]] - `method` [EXTRACTED]
- [[.score_rollout()_1]] - `method` [EXTRACTED]
- [[PerplexityScorer]] - `calls` [EXTRACTED]
- [[Rollout_8]] - `uses` [INFERRED]
- [[Rollout (shared record)]] - `references` [EXTRACTED]
- [[Score_7]] - `uses` [INFERRED]
- [[Score (shared record)]] - `references` [EXTRACTED]
- [[Text-only detector scores a rollout by its instruction's perplexity.      Args]] - `rationale_for` [EXTRACTED]
- [[_perplexity_to_score]] - `calls` [EXTRACTED]
- [[goal_agnostic_anomaly_score()]] - `semantically_similar_to` [INFERRED]
- [[perplexity.py]] - `contains` [EXTRACTED]
- [[test_calibrates_identically_through_shared_calibrate()]] - `calls` [INFERRED]
- [[test_empty_rollout_raises()]] - `calls` [INFERRED]
- [[test_higher_perplexity_maps_to_higher_score()]] - `calls` [INFERRED]
- [[test_perplexity.py]] - `references` [EXTRACTED]
- [[test_perplexity_one_maps_to_zero_and_below_one_clamps()]] - `calls` [INFERRED]
- [[test_real_backend_plugs_into_filter_but_errors_when_used()]] - `calls` [INFERRED]
- [[test_score_rollout_returns_single_unit_interval_score()]] - `calls` [INFERRED]
- [[test_scores_the_operational_instruction_not_the_trusted_goal()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter