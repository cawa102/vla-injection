# OpenVLA Backbone Attack Transfer Notes

> **Status:** Research scaffold for author review. Created 2026-06-24 after the
> supervisor discussion about shifting the project toward offense-side VLA security.
> Do not cite a claim from this note unless the linked primary paper has been read
> and the specific claim is verified.

## 1. Architectural premise

OpenVLA-7B builds on a Llama 2 language model and a visual encoder that fuses
pretrained DINOv2 and SigLIP features (`arXiv:2406.09246`). That makes three
pre-existing attack literatures relevant:

- Llama 2 / LLM attacks: adversarial suffixes, prompt injection, jailbreaks,
  subversive fine-tuning, language-side backdoors.
- DINOv2 / vision-foundation-model attacks: feature-space perturbations,
  OOD/ID representation flips, patch-level attacks.
- SigLIP / CLIP-family attacks: image-text alignment attacks, typographic
  attacks, universal adversarial perturbations, CLIP backdoors.

The key caveat is that OpenVLA is not a plain Llama 2, DINOv2, or SigLIP model.
Robot fine-tuning and action-token decoding change the objective. So the right
claim is not "backbone attacks transfer directly"; it is "backbone attack
mechanisms are plausible VLA attack templates if the loss is rewritten against
action reachability, trajectory deviation, or instruction-action misalignment."

## 2. Ordered threat list for OpenVLA

### 1. Llama-side adversarial suffix / GCG attacks

**Source lineage.** GCG-style attacks on aligned LLMs find adversarial suffixes
with greedy gradient search and transfer to LLaMA-2-Chat and other models
(`arXiv:2307.15043`). RoboGCG already adapts this idea to VLAs and shows that a
textual suffix applied once at rollout start can provide broad action-space
reachability in common VLA policies (`arXiv:2506.03350`).

**Why it matters for OpenVLA.** OpenVLA emits actions through the Llama-token
generation pathway, so the language backbone and action-token head share the same
autoregressive control surface. This is the most direct and already VLA-proven
route.

**Research use.** Strong primary lane. Natural extensions: quantized-surrogate
GCG transfer to bf16 OpenVLA; semantic target construction rather than only
low-level action reachability; transfer across LIBERO fine-tunes.

### 2. CLIP/SigLIP universal visual perturbations

**Source lineage.** Vision-language pretrained models are vulnerable to universal
image perturbations that transfer across downstream tasks and VLP architectures
(`arXiv:2405.05524`). X-Transfer reports highly transferable UAPs across CLIP
encoders and downstream VLMs (`arXiv:2505.05528`).

**Why it matters for OpenVLA.** SigLIP is CLIP-like: image and text are aligned in
a shared representation space, and OpenVLA uses SigLIP visual features as part of
its fused visual encoder. A perturbation that disrupts image-text alignment may
misground the instruction before action decoding.

**Research use.** High-value secondary lane. Convert VLP losses to VLA losses:
maximize action deviation, target wrong object, or reduce similarity between the
scene feature and the trusted instruction.

### 3. DINOv2 feature-space attacks

**Source lineage.** Foundational vision-model attacks manipulate deep feature
representations of CLIP, DINOv2, and similar encoders, including black-box
transfer across model families (`arXiv:2308.14597`). DINOv2-based anomaly
detectors can also be degraded by imperceptible gradient perturbations that flip
nearest-neighbor feature relations (`arXiv:2510.13643`).

**Why it matters for OpenVLA.** DINOv2 contributes spatial/object-centric visual
features. Feature-space perturbations against DINOv2 are plausible attacks
against the object localization and scene-understanding part of OpenVLA.

**Research use.** Good visual-side lane, especially for wrong-object and
wrong-region manipulation. More work than RoboGCG because the loss must be
connected to LIBERO action outcomes rather than image classification/OOD labels.

### 4. Typographic / visual prompt injection attacks

**Source lineage.** CLIP-family models can be manipulated by text inserted into
images; recent work localizes typographic circuits in CLIP vision encoders and
studies defenses against text-in-image attacks (`arXiv:2508.20570`).

**Why it matters for OpenVLA.** A robot camera can observe labels, stickers,
screens, paper notes, or object packaging. If SigLIP encodes visible text
strongly, text-in-scene can become an indirect instruction channel even when the
operator's natural-language instruction is clean.

**Research use.** Very relevant threat model, but likely needs careful target
definition: visible words should cause action change, not merely visual
misclassification. Could be tested cheaply in LIBERO by rendering text patches or
placing textured planes.

### 5. Universal / transferable VLA patch attacks

**Source lineage.** VLA-specific visual patch attacks now exist. VLA-Hijack
targets the shared visual self-localization/proprioception requirement and
reports transfer across OpenVLA, UniVLA, and CronusVLA (`arXiv:2605.28083`).
UPA-RFAS studies universal transferable physical patches across VLA models
(`arXiv:2511.21192`). VLA-Fool combines textual, visual, and cross-modal
misalignment attacks on fine-tuned OpenVLA in LIBERO (`arXiv:2511.16203`).

**Why it matters for OpenVLA.** These are not merely backbone attacks; they are
already VLA attacks. They support the professor's visual-channel suggestion.

**Research use.** Good if the project pivots to visual attacks, but novelty risk
is higher because this lane is active and crowded in 2025-2026.

### 6. CLIP/SigLIP backdoors and prompt-tuning backdoors

**Source lineage.** BadCLIP shows trigger-aware prompt learning can implant
high-ASR CLIP backdoors while preserving clean accuracy (`arXiv:2311.16194`).
Later work studies model-level backdoor detection in prompt-tuned CLIP models
(`arXiv:2604.09101`, `arXiv:2512.00343`).

**Why it matters for OpenVLA.** If OpenVLA is adapted via prompt tuning, adapter
tuning, or third-party fine-tunes, backdoors can sit in the vision-language
alignment path without changing the base encoders. VLA-specific versions already
exist through BadVLA and TabVLA.

**Research use.** Relevant but heavier. It shifts the project from inference-time
attack reproduction to poisoning / supply-chain evaluation.

### 7. Llama-side subversive fine-tuning / weight-level backdoors

**Source lineage.** LoRA fine-tuning can remove safety behavior from Llama 2-Chat
with low budget (`arXiv:2310.20624`), and BadEdit shows lightweight model editing
can inject LLM backdoors robust to later fine-tuning (`arXiv:2403.13355`).

**Why it matters for OpenVLA.** OpenVLA inherits a Llama 2 backbone and is commonly
fine-tuned. A malicious fine-tune or edited checkpoint could alter instruction
following or action-token preferences.

**Research use.** Related work / future lane unless the dissertation pivots to
trojaned checkpoints. VLA-specific backdoor papers are more directly relevant
than generic Llama attacks.

### 8. CoT / reasoning-patch attacks

**Source lineage.** TRAP uses adversarial patches to corrupt CoT reasoning in
reasoning-enabled VLA models and induce targeted behavior without modifying the
user instruction (`arXiv:2603.23117`).

**Why it matters for OpenVLA.** Current OpenVLA-7B is a vanilla non-CoT VLA, so
TRAP is not directly applicable. It matters if the study expands to reasoning
VLAs or if an OpenVLA-derived system adds explicit planning tokens.

**Research use.** Future/generalization arm, not a smooth immediate pivot.

## 3. Immediate recommendation

The smoothest offense-side pivot remains language-side:

> Quantized-surrogate RoboGCG transfer: optimize GCG on 4-bit/8-bit OpenVLA and
> measure transfer to bf16 OpenVLA.

This reuses the working RoboGCG/OpenVLA/LIBERO setup and turns the research
question into an attacker capability question. The visual-backbone list above is
still valuable: it provides a ranked secondary lane if the supervisor wants a
stronger visual-channel story. The best visual follow-up would be a small
SigLIP/DINOv2 feature-attack pilot against wrong-object / wrong-region action
selection, not a full new patch-attack framework.

## 4. Primary sources to verify before citing

- OpenVLA architecture: https://arxiv.org/abs/2406.09246
- Llama 2: https://arxiv.org/abs/2307.09288
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
- GCG on LLMs: https://arxiv.org/abs/2307.15043
- RoboGCG: https://arxiv.org/abs/2506.03350
- Foundational vision-model attacks: https://arxiv.org/abs/2308.14597
- VLP universal perturbations: https://arxiv.org/abs/2405.05524
- X-Transfer CLIP UAPs: https://arxiv.org/abs/2505.05528
- Typographic CLIP attacks: https://arxiv.org/abs/2508.20570
- DINOv2 adversarial anomaly detection: https://arxiv.org/abs/2510.13643
- VLA-Fool: https://arxiv.org/abs/2511.16203
- VLA-Hijack: https://arxiv.org/abs/2605.28083
- UPA-RFAS VLA patch attack: https://arxiv.org/abs/2511.21192
- BadVLA: https://arxiv.org/abs/2505.16640
- TabVLA: https://arxiv.org/abs/2510.10932
- TRAP: https://arxiv.org/abs/2603.23117
