# Plan Review Log: M1 Attack-Target Redesign

Act 1 (grill) — intent locked with the user via the 2026-07-03 AskUserQuestion decision (two-tier target: RoboGCG-clean anchor + policy-derived semantic redirect; report both success notions). MAX_ROUNDS=3.

Documents under review:
- `docs/plans/2026-07-03-m1-attack-target-redesign.md` (implementation plan)
- `docs/core/execution-playbook.md` (§1 status block + D2 decision row)

Codex runs read-only every round; Claude is final arbiter on each REVISE.

## Round 1 — Codex

**Findings**

- Tier B ASR is partly circular with L2: defining success as EE within the distractor engagement radius reuses the same primitive that `ConsistencyMetricA` uses for distractor engagement, so L2 separation can become near-tautological even if `SchemaA` is benign-pinned. Fix: pre-register a detector-independent Tier B success label, or report a P2-ablated L2 score alongside the normal detector.

- “On-manifold by construction” is overstated: `a* = π(image, adv_instruction)` is only a single-frame policy action, not evidence that the suffix can cause a closed-loop wrong-object redirect as frames change. Fix: reword to “single-frame token-manifold target” and make clean `adv_instruction` closed-loop distractor reach mandatory before including a Tier B task.

- Task 4’s interface is underspecified/wrong for token ids: OpenVLA `predict_action` returns a continuous 7-DoF action, while GCG needs teacher-forced action token ids; `ActionCodec` is decode-only. Fix: either use generation/logits to capture the seven action tokens directly, or add a tested encode path from normalized actions to token ids.

- There is a real vocab-size trap: action decoding uses `model.config.text_config.vocab_size - model.config.pad_to_multiple_of`, while `OpenVlaGcgTarget.vocab_size` is `processor.tokenizer.vocab_size` for suffix search. Fix: name and log `action_vocab_size` separately from `suffix_vocab_size`, and validate semantic target ids only against the action range.

- The plan says to reuse `state_libero.extract_ee_pos` / `extract_object_poses` for `rollout_asr_world`, but those parse raw LIBERO obs keys, not serialized `RolloutStep.privileged_state`. Fix: parse rollout states through `SyntheticStateAdapter` / `PrivilegedState`, matching `geometry_stats` and `ConsistencyMetricA`.

- The two ASR frames will be conflated by the current gate unless the schema changes more aggressively: `AttackUnitRecord` has one `rollout_asr_reached`, and `m1_verdict` aggregates all attack records into one ASR rate. Fix: make `target_tier` and `asr_frame` required, enforce homogeneous records in `m1_gate`, or group metrics by tier.

- The resume guard does not yet include the proposed tier or ASR-frame fields, so a semantic run can accidentally resume an anchor directory if other header fields match. Fix: add `target_tier`, registry hash, adversary registry path/version, and ASR frame to `_RESUME_KEYS`.

- Task 2’s `positions` semantics are ambiguous under the causal shift: current `target_span_argmax_matches` compares `labels[1:]`, but the plan says `positions=[0..5]` without specifying whether those are raw labels or target-action positions. Fix: define positions relative to the seven action target ids after shift, and test against `labels_slice=[ignore]+target_ids`.

- The RoboGCG comparison is not fully supported by the repo’s verified facts: local references record >90% on Goal/Object/Spatial and 77.4% on Libero-10, not a plain “79% on OpenVLA”. Fix: replace the 79% sentence with exact suite-level numbers or show the calculation and source.

- “Identical to RoboGCG” is too strong: repo facts record RoboGCG’s ASCII / retokenization-safe filtering, while this GCG samples arbitrary tokenizer ids. Fix: either implement the same candidate filter or explicitly log this as a deviation from RoboGCG.

- The anchor full sweep is internally inconsistent: the plan cites RoboGCG `1792 = 7×256`, but its proposed `anchor_action_space(max_mag_only=False)` iterates only six motion dims. Fix: either include dim 6 in the non-max sweep and exclude it only from scoring, or state that this is a six-motion-dim variant.

- Literal `±1` anchor targets are not actual OpenVLA bin centers; `target_action_ids_for` will map them to nearest edge bins. Fix: construct anchors from `_BIN_CENTERS[0]` and `_BIN_CENTERS[-1]` and describe them as edge-bin max magnitude.

- The separation guard is too weak: object-object distance `> engagement_radius + grasp_radius` does not rule out the initial EE already being near the distractor, benign paths passing through the distractor region, or dynamic object motion. Fix: add per-unit checks for initial EE-distractor distance and benign-rollout distractor-window false positives before accepting a Tier B target.

- For “pick up the distractor”, ASR as EE-near-object is only approach, not manipulation; gripper exclusion makes semantic wrong-object success too generous. Fix: split Tier B into `approach_asr` and `manipulation_asr`, and require gripper close/object displacement for pick/place claims.

- The registry keyed only by `(suite, task_name)` is brittle because config uses symbolic `task_0` while runtime also has LIBERO `task.name`, and object availability can vary by init state. Fix: store canonical suite, task index, LIBERO task name, scene/init-state constraints, and validate `distractor_object` after reset for every attacked unit.

- The D2 row is contradictory in status: it says the semantic arm was M1-gated and “DECIDED”, while the rebalance is “proposed, supervisor sign-off pending.” Fix: mark the rebalance as `PROPOSED` consistently and block implementation past pilot scaffolding until sign-off.

VERDICT: REVISE

### Claude's response (Round 1)

Final arbiter: **all 16 findings accepted**; one scoped down. Changes applied to `docs/plans/2026-07-03-m1-attack-target-redesign.md` and `docs/core/execution-playbook.md`:

- **Tier-B/L2 circularity (critical):** added a **Tier-B detector-independence guard** — world-frame ASR reuses the detector's P2 primitive, so L2 TPR on Tier B must be P2-ablated (or use a detector-independent label). Reflected in Task 5 notes + Guards.
- **"On-manifold" overstated:** reworded to "single-frame token-manifold target"; clean-adversary-rollout reach + benign-FP floor **promoted from stretch to mandatory** (Task 6, Guards).
- **Task 4 token-ids:** rewritten to capture the 7 action token ids from **greedy generation** (argmax over action positions); explicitly **do NOT use `predict_action`** (continuous) and noted `ActionCodec` is **decode-only** (verified).
- **vocab_size trap:** added an `action_vocab_size` vs `suffix_vocab_size` disambiguation guard; target ids validated against the action range (Tasks 4, 7).
- **World-frame state access:** Task 5 now reads `SyntheticStateAdapter`/`PrivilegedState` (like `geometry_stats`), not the raw-obs extractors.
- **Two ASR frames conflation:** `target_tier`/`asr_frame` now **required**; `m1_verdict` must not aggregate across tiers; `_RESUME_KEYS` extended with tier/asr_frame/registry-hash.
- **Task 2 positions:** defined relative to the 7 action target ids after the causal shift; test vs `labels_slice=[ignore]+target_ids`.
- **RoboGCG numbers:** replaced "79% on OpenVLA" with the repo-verified >90% (Goal/Object/Spatial) / 77.4% (LIBERO-10), reached in 30–110 steps; attributed the 79% to the AttackVLA survey. Applied in plan + playbook.
- **"Identical to RoboGCG":** softened to "match, except candidate-id filtering" (logged deviation).
- **Anchor sweep:** clarified as a **6-motion-dim variant** (gripper excluded); anchors use **edge bin centres** `_BIN_CENTERS[0]/[-1]`, not literal ±1.
- **Separation guard too weak:** added initial-EE-distractor distance + benign-rollout distractor-window FP check (Task 6).
- **Approach ≠ manipulation:** Tier-B ASR split into `approach_asr` / `manipulation_asr`; no "picked the wrong object" claim from approach alone.
- **Registry brittleness:** canonical key (suite, task_index, LIBERO task.name); per-unit distractor validation after reset.
- **D2 status contradiction:** rewritten so the two-tier rebalance is consistently **PROPOSED (sign-off pending)**, original decision noted as the superseded-pending baseline.

**Scoped (not fully implemented):** candidate-id ASCII/special filtering (finding 10) — recorded as a **logged deviation** + optional future `filter_ids` pass (YAGNI until shown to matter), rather than mandatory now.

## Round 2 — Codex

**Prior Findings Check**

Most prior blockers are now addressed in the plan: the RoboGCG ASR statement is corrected, `action_vocab_size` vs suffix vocab is called out, `predict_action` is no longer proposed for token-id capture, `SyntheticStateAdapter` is specified for rollout state parsing, ASR frame/tier fields and resume guards are required, target-position indexing is clarified, edge-bin anchors replace literal `±1`, Tier B clean-benign and clean-adversary validity checks are mandatory, registry runtime validation is added, and D2 is now marked proposed/sign-off pending.

**Still Material**

- [execution-playbook.md](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/core/execution-playbook.md:79) still says RoboGCG parity is “identical” and “the sole difference is the target,” contradicting the revised plan’s candidate-filter deviation. Fix: update the status block to match the plan: “matches except ASCII/retokenization filtering; target is the decisive difference.”

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:28) requires P2-ablated L2 reporting, but no task/interface/test/record field actually implements an ablated scorer. Fix: add an explicit subtask to implement/report `metric_a_p2_ablated_per_step` or a primitive-mask scorer, with `m1_gate` grouping.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:24) introduces `manipulation_asr`, but Task 5 only defines `rollout_asr_world` for approach. Fix: add a concrete `rollout_manipulation_asr` definition, thresholds, state sources, and tests, or downgrade manipulation to non-claiming logged diagnostic.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:152) still underspecifies semantic token capture: “greedy generation” and “argmax over action-token positions” need exact prompt/template/span mechanics. Fix: specify the exact `generate`/forward call and require a faithfulness test that `codec.decode(captured_ids)` matches `predict_action(..., do_sample=False)` within one bin.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:113) updates `AdversarySpec`, but the JSON example at lines 117-124 still lacks `task_index`, `libero_task_name`, and init-state constraints. Fix: update the config schema example to match the dataclass and tests.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:199) says separation checks are “per calib task,” but availability/separation can depend on seed/init state and the post-settle frame. Fix: require checks per attacked unit after `reset_and_settle`, not just per task.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:98) logs the RoboGCG candidate-filter deviation but keeps calling Tier A “RoboGCG-clean/prior-comparable.” Fix: either implement the ASCII/retokenization filter for the anchor runs or rename the claim to “RoboGCG-target-clean, search-space deviation logged.”

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:268) still says “RoboGCG itself fails ~21%,” which is the stale ~79% framing. Fix: replace with suite-specific failure wording or drop the percentage.

- [plan](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:7) says the existing closed-loop harness is unchanged, but Task 7 changes driver schema, resume keys, m1 aggregation, and Tier B scoring. Fix: change this to “same GCG core; driver/record schema extended.”

VERDICT: REVISE

### Claude's response (Round 2)

All 9 residual findings accepted (most were consistency gaps my Round-1 edits introduced):
- playbook status block: "identical" → "match, except candidate-id filtering" (now consistent with the plan).
- **P2-ablated L2 scorer** given a real implementing subtask (Task 5: `ConsistencyMetricA.score_rollout(..., ablate_primitives=...)` + `metric_a_p2_ablated_per_step`, tested) — the guard is no longer orphaned.
- **`manipulation_asr` downgraded to a logged diagnostic** (headline = `approach_asr`); full manipulation scorer noted as future work.
- Task 4 token capture: exact template + trailing-7-position slice specified; added a **faithfulness test** (`codec.decode(captured_ids)` ≈ `predict_action(do_sample=False)` within 1 bin — predict_action as the check, not the source).
- JSON config example updated to match the new `AdversarySpec` (`task_index`, `libero_task_name`, `init_state_constraints`).
- Separation checks moved to **per attacked unit after `reset_and_settle`** (not per task).
- Tier A relabelled "RoboGCG-*target*-clean (search-space deviation logged)".
- Stale "~21%" replaced with suite-specific wording (77.4% LIBERO-10).
- Architecture line corrected: "same GCG core; driver/record schema extended."

## Round 3 — Codex

Most Round-2 issues are now addressed: the playbook no longer claims exact RoboGCG identity, P2 ablation is a real Task-5 subtask, manipulation is downgraded, registry schema is updated, separation is per attacked unit after `reset_and_settle`, stale ~21% wording is fixed, and the architecture/record-schema language is now honest.

One material blocker remains in Task 4. [The semantic target builder](/home/40473058@eeecs.qub.ac.uk/vla-injection/docs/plans/2026-07-03-m1-attack-target-redesign.md:158) says to use “greedy generation” but also “one no-grad forward” and read the trailing 7 action positions using the same `_target_span_ce_torch`/`reached` slice. That slice only exists after target/action ids have already been appended to the text input; with just the prompt, a single causal forward gives logits for the next token only, not seven future action-token positions. Fix: specify `model.generate(max_new_tokens=7, do_sample=False)` or an explicit autoregressive 7-step loop, capture the seven generated ids, then run the `codec.decode(captured_ids)` vs `predict_action(..., do_sample=False)` faithfulness test.

Minor implementation note while revising: Task 5’s proposed `score_rollout(..., *, ablate_primitives=...)` should preserve the existing `trusted_goal` argument to avoid breaking current callers.

VERDICT: REVISE

### Claude's response (Round 3)

Both findings accepted (convergence monotonic: 16 → 9 → 2, zero rejected):
- **Task 4 (material):** corrected the token-capture mechanics — a prompt-only forward yields only next-token logits, so use `model.generate(max_new_tokens=7, do_sample=False)` (or a 7-step greedy loop), then the faithfulness check. My R2 "one no-grad forward + trailing slice" was wrong (that slice only exists under teacher forcing).
- **Task 5 (minor):** `score_rollout` keeps the existing `trusted_goal` positional arg; `ablate_primitives` is keyword-only after it — no caller breakage.

MAX_ROUNDS was 3, but every finding across all rounds was accepted (no disagreements) and the two Round-3 items were non-contentious correctness fixes — so this is clean convergence, not deadlock. Extending by one confirmation round (→4) to capture the final verdict.

## Round 4 (confirmation) — Codex

Confirmed. The Task 4 mechanics now correctly require `model.generate(..., max_new_tokens=7, do_sample=False)` or an explicit greedy autoregressive loop, and explicitly reject the invalid single-forward/trailing-slice approach for unknown future tokens.

Task 5 also preserves the existing `trusted_goal` positional argument while adding `ablate_primitives` keyword-only, so the scorer extension is compatible with current callers. I do not see remaining material blockers for handing this to an implementation session.

VERDICT: APPROVED
---

**Resolution: APPROVED after 4 rounds** (findings 16 → 9 → 2 → 0, every finding accepted, none rejected). Documents ready for an implementing session, pending supervisor sign-off on the D2 rebalance.
