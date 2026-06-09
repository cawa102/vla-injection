---
source_file: "src/evasion_tax/baselines/perplexity.py"
type: "rationale"
community: "L0 Perplexity Filter"
location: "L71"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/L0_Perplexity_Filter
---

# The real LM-perplexity backend — deferred to the GPU node (raises if used here).

## Connections
- [[RealPerplexityScorer]] - `rationale_for` [EXTRACTED]
- [[perplexity.py]] - `defined_in` [EXTRACTED]

#graphify/rationale #graphify/EXTRACTED #community/L0_Perplexity_Filter