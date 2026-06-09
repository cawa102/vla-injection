---
source_file: "tests/evasion_tax/baselines/test_perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L107"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# test_calibrates_identically_through_shared_calibrate()

## Connections
- [[MockPerplexityScorer]] - `calls` [INFERRED]
- [[PerplexityFilter]] - `calls` [INFERRED]
- [[calibrate()]] - `calls` [INFERRED]
- [[make_rollout()]] - `calls` [EXTRACTED]
- [[rollout_fires()]] - `calls` [INFERRED]
- [[test_perplexity.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter