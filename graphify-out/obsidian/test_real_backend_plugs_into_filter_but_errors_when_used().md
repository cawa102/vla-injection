---
source_file: "tests/evasion_tax/baselines/test_perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L151"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# test_real_backend_plugs_into_filter_but_errors_when_used()

## Connections
- [[PerplexityFilter]] - `calls` [INFERRED]
- [[RealPerplexityScorer]] - `calls` [INFERRED]
- [[make_rollout()]] - `calls` [EXTRACTED]
- [[test_perplexity.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter