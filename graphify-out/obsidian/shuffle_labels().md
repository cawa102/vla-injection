---
source_file: "src/evasion_tax/metric/probe_confounds.py"
type: "code"
community: "L1 Internal Probe"
location: "L27"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L1_Internal_Probe
---

# shuffle_labels()

## Connections
- [[Return a deterministic permutation of ``labels`` (multiset preserved).      The]] - `rationale_for` [EXTRACTED]
- [[probe_confounds.py]] - `contains` [EXTRACTED]
- [[test_label_shuffle_control_collapses_probe_to_chance()]] - `calls` [INFERRED]
- [[test_probe_confounds.py]] - `references` [EXTRACTED]
- [[test_shuffle_labels_changes_order_and_varies_with_seed()]] - `calls` [INFERRED]
- [[test_shuffle_labels_does_not_mutate_input()]] - `calls` [INFERRED]
- [[test_shuffle_labels_preserves_multiset_and_is_deterministic()]] - `calls` [INFERRED]

#graphify/code #graphify/INFERRED #community/L1_Internal_Probe