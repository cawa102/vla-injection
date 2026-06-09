---
source_file: "tests/evasion_tax/baselines/test_perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L75"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# test_perplexity_one_maps_to_zero_and_below_one_clamps()

## Connections
- [[MockPerplexityScorer]] - `calls` [INFERRED]
- [[PerplexityFilter]] - `calls` [INFERRED]
- [[make_rollout()]] - `calls` [EXTRACTED]
- [[test_perplexity.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter