---
source_file: "src/evasion_tax/baselines/perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L53"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# MockPerplexityScorer

## Connections
- [[.__init__()]] - `method` [EXTRACTED]
- [[.score_perplexity()_1]] - `method` [EXTRACTED]
- [[Deterministic perplexity backend for tests (no LM).      Args         table Op]] - `rationale_for` [EXTRACTED]
- [[PerplexityScorer]] - `implements` [INFERRED]
- [[Rollout_8]] - `uses` [INFERRED]
- [[Score_7]] - `uses` [INFERRED]
- [[_heuristic_perplexity]] - `calls` [EXTRACTED]
- [[perplexity.py]] - `contains` [EXTRACTED]
- [[test_calibrates_identically_through_shared_calibrate()]] - `calls` [INFERRED]
- [[test_empty_rollout_raises()]] - `calls` [INFERRED]
- [[test_higher_perplexity_maps_to_higher_score()]] - `calls` [INFERRED]
- [[test_mock_heuristic_empty_instruction_is_minimal_perplexity()]] - `calls` [INFERRED]
- [[test_mock_heuristic_rates_symbol_heavy_text_more_perplexing_than_english()]] - `calls` [INFERRED]
- [[test_perplexity.py]] - `references` [EXTRACTED]
- [[test_perplexity_one_maps_to_zero_and_below_one_clamps()]] - `calls` [INFERRED]
- [[test_score_rollout_returns_single_unit_interval_score()]] - `calls` [INFERRED]
- [[test_scores_the_operational_instruction_not_the_trusted_goal()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter