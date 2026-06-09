---
source_file: "src/evasion_tax/baselines/perplexity.py"
type: "code"
community: "L0 Perplexity Filter"
location: "L70"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L0_Perplexity_Filter
---

# RealPerplexityScorer

## Connections
- [[.score_perplexity()_2]] - `method` [EXTRACTED]
- [[PerplexityScorer]] - `implements` [INFERRED]
- [[Rollout_8]] - `uses` [INFERRED]
- [[Score_7]] - `uses` [INFERRED]
- [[The real LM-perplexity backend — deferred to the GPU node (raises if used here).]] - `rationale_for` [EXTRACTED]
- [[perplexity.py]] - `contains` [EXTRACTED]
- [[test_perplexity.py]] - `references` [EXTRACTED]
- [[test_real_backend_is_gpu_stub()]] - `calls` [INFERRED]
- [[test_real_backend_plugs_into_filter_but_errors_when_used()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/L0_Perplexity_Filter