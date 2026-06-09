---
source_file: "src/evasion_tax/metric/probe_confounds.py"
type: "code"
community: "L1 Internal Probe"
location: "L44"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/L1_Internal_Probe
---

# probe_auc()

## Connections
- [[ActivationFeatures]] - `references` [EXTRACTED]
- [[InternalProbe]] - `references` [EXTRACTED]
- [[ROC-AUC of ``probe`` over labelled ``features`` (benign=0, injected=1).      Spl]] - `rationale_for` [EXTRACTED]
- [[probe_confounds.py]] - `contains` [EXTRACTED]
- [[roc_auc()]] - `calls` [INFERRED]
- [[test_label_shuffle_control_collapses_probe_to_chance()]] - `calls` [INFERRED]
- [[test_probe_auc_is_high_for_a_separable_probe()]] - `calls` [INFERRED]
- [[test_probe_auc_rejects_length_mismatch()]] - `calls` [INFERRED]
- [[test_probe_auc_rejects_single_class_labels()]] - `calls` [INFERRED]
- [[test_probe_confounds.py]] - `references` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/L1_Internal_Probe