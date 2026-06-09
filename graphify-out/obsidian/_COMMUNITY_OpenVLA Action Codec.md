---
type: community
members: 41
---

# OpenVLA Action Codec

**Members:** 41 nodes

## Members
- [[.__post_init__()_9]] - code - src/evasion_tax/policy/action_codec.py
- [[.action_dim()]] - code - src/evasion_tax/policy/action_codec.py
- [[.bin_centers()]] - code - src/evasion_tax/policy/action_codec.py
- [[.bin_to_norm()]] - code - src/evasion_tax/policy/action_codec.py
- [[.bins()]] - code - src/evasion_tax/policy/action_codec.py
- [[.decode()]] - code - src/evasion_tax/policy/action_codec.py
- [[.from_stats()]] - code - src/evasion_tax/policy/action_codec.py
- [[.token_to_bin()]] - code - src/evasion_tax/policy/action_codec.py
- [[.unnormalize()]] - code - src/evasion_tax/policy/action_codec.py
- [[ActionCodec]] - code - src/evasion_tax/policy/action_codec.py
- [[Build a codec from a fetched ``dataset_statistics`` mapping.          Mirrors Op]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Decode OpenVLA discrete action tokens to continuous actions.      Immutable (cod]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Decode a full action's token ids to a continuous action vector.          Args]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Dimensionality of the action space (= number of quantile entries).]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Independent re-statement of the verified source decode+un-normalise.]] - rationale - tests/evasion_tax/policy/test_action_codec.py
- [[Map a single action token id to a bin-centre index in ``0, n_bins-2``.]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Mirror of OpenVLA ActionTokenizer.__call__ (normalized action - token id).]] - rationale - tests/evasion_tax/policy/test_action_codec.py
- [[OpenVLA action codec discrete action token ids - continuous 7-DoF (Task 3).  T]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Return the normalised value (bin centre, in ``-1, 1``) for a bin index.]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Tests for the OpenVLA action codec (Task 3).  Every numeric expectation here is]] - rationale - tests/evasion_tax/policy/test_action_codec.py
- [[The ``n_bins - 1`` bin centres (what de-tokenised tokens map to).]] - rationale - src/evasion_tax/policy/action_codec.py
- [[The ``n_bins`` uniform bin edges over ``-1, 1``.]] - rationale - src/evasion_tax/policy/action_codec.py
- [[Un-normalise per-dim normalised actions using q01q99 under ``mask``.          `]] - rationale - src/evasion_tax/policy/action_codec.py
- [[_reference_decode()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[_tokenize_norm()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[action_codec.py]] - code - src/evasion_tax/policy/action_codec.py
- [[ndarray_7]] - code - src/evasion_tax/policy/action_codec.py
- [[test_action_codec.py]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_bin_centers_has_n_bins_minus_one_entries_strictly_increasing_inside_range()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_bin_to_norm_returns_the_bin_center()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_codec_is_immutable()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_constructor_rejects_mismatched_stat_lengths()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_constructor_rejects_q01_above_q99()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_decode_bin_round_trip_recovers_normalized_within_one_bin_halfwidth()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_decode_full_7dof_pipeline_matches_openvla_source_formula()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_decode_rejects_token_sequence_of_wrong_length()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_from_stats_extracts_quantiles_and_defaults_mask_all_true()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_from_stats_preserves_explicit_mask()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_from_stats_unknown_unnorm_key_raises()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_token_to_bin_applies_vocab_offset_and_off_by_one_clip()]] - code - tests/evasion_tax/policy/test_action_codec.py
- [[test_unnormalize_uses_quantiles_on_masked_dims_and_passes_through_unmasked()]] - code - tests/evasion_tax/policy/test_action_codec.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/OpenVLA_Action_Codec
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Stats Provenance]]

## Top bridge nodes
- [[action_codec.py]] - degree 22, connects to 1 community