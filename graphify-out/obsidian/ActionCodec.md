---
source_file: "src/evasion_tax/policy/action_codec.py"
type: "code"
community: "OpenVLA Action Codec"
location: "L40"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/OpenVLA_Action_Codec
---

# ActionCodec

## Connections
- [[.__post_init__()_9]] - `method` [EXTRACTED]
- [[.action_dim()]] - `method` [EXTRACTED]
- [[.bin_centers()]] - `method` [EXTRACTED]
- [[.bin_to_norm()]] - `method` [EXTRACTED]
- [[.bins()]] - `method` [EXTRACTED]
- [[.decode()]] - `method` [EXTRACTED]
- [[.from_stats()]] - `references` [EXTRACTED]
- [[.token_to_bin()]] - `method` [EXTRACTED]
- [[.unnormalize()]] - `method` [EXTRACTED]
- [[Decode OpenVLA discrete action tokens to continuous actions.      Immutable (cod]] - `rationale_for` [EXTRACTED]
- [[action_codec.py]] - `contains` [EXTRACTED]
- [[test_action_codec.py]] - `references` [EXTRACTED]
- [[test_bin_centers_has_n_bins_minus_one_entries_strictly_increasing_inside_range()]] - `calls` [INFERRED]
- [[test_bin_to_norm_returns_the_bin_center()]] - `calls` [INFERRED]
- [[test_codec_is_immutable()]] - `calls` [INFERRED]
- [[test_constructor_rejects_mismatched_stat_lengths()]] - `calls` [INFERRED]
- [[test_constructor_rejects_q01_above_q99()]] - `calls` [INFERRED]
- [[test_decode_bin_round_trip_recovers_normalized_within_one_bin_halfwidth()]] - `calls` [INFERRED]
- [[test_decode_full_7dof_pipeline_matches_openvla_source_formula()]] - `calls` [INFERRED]
- [[test_decode_rejects_token_sequence_of_wrong_length()]] - `calls` [INFERRED]
- [[test_token_to_bin_applies_vocab_offset_and_off_by_one_clip()]] - `calls` [INFERRED]
- [[test_unnormalize_uses_quantiles_on_masked_dims_and_passes_through_unmasked()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/OpenVLA_Action_Codec