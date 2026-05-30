# CLAUDE.md

Guidance for Claude Code in this directory. These take precedence over default behaviour.
**Caution over speed.** The four principles are general; the **Customization** section adapts them to this project.

---

## The Four Principles

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Customization (this project)

**Project.** MSc Cyber Security & AI individual research project on **VLA-model security — integrity focus**
(does the perception→reasoning→action pipeline produce correct, untampered actions; how is it subverted or
defended). A graded, single-author dissertation: reproducibility and defensible claims matter more than speed.

**Current theme — agreed 2026-05-30, now in *Design*.** **T7 — *Goal-action consistency detection for
instruction-injected VLA policies under calibrated false-positive constraints.*** A *measurement* study of an
attacker-aware, FP-calibrated detector — **not** a new universal defense. Base model **OpenVLA-7B**,
**simulation only (LIBERO)**. Working notes: `docs/plans/t7-goal-action-consistency-detector.md`; landscape:
`docs/lit-review/`. **Don't write experiment code until a plan is agreed in `docs/plans/`.**

**Phases — know which one a task belongs to; don't skip ahead.**
Scope → Literature review → Design → Implement → Run & analyse → Write up.

**Reproducibility (non-negotiable).** Pin & record all seeds; capture the exact environment; record
data/checkpoint provenance (source, hash, date, licence); log each run to a timestamped **write-once**
`results/`; change **one variable at a time**; figures regenerable from logged data by a script; report
negative results — don't drop them.

**Academic integrity.** No fabricated sources — mark `[CITATION NEEDED]` and verify (primary papers /
Context7) before asserting; distinguish "established result" from "my experiment showed"; generated prose is
a draft for the author to rewrite; attribute borrowed code/data/ideas and respect licences.

**Security-research ethics.** Attack code (poisoning, perturbation, trojans) is built only for **contained,
authorised, defensive** evaluation. Quarantine poisoned data / trojaned checkpoints under
`artifacts/untrusted/`; never auto-run untrusted checkpoints; follow the institution's ethics process.

**Repo conventions.** `docs/` · `src/` · `configs/` · `data/` (gitignored) · `results/` (write-once) ·
`scripts/` · `artifacts/` (gitignored) — create a directory only when there's something to put in it. Never
commit datasets, checkpoints, secrets, or PII. **Git not yet initialised — confirm before `git init` or any
first commit.**

**VLA vocabulary.** VLA = vision + language instruction → robot **actions**; a wrong output is a wrong
*physical action*. Integrity threat channels in scope: visual adversarial perturbation, prompt/instruction
injection, data/backdoor poisoning & trojaned weights, action-space manipulation, supply-chain tampering.
Confidentiality (extraction) and availability (DoS) are secondary unless they serve the integrity story.

**Global rules.** `~/.claude/rules/*` (coding style, testing, security) remain in force; this file adds the
research-project layer on top.

---

## How to Know It's Working

Fewer unnecessary diffs; fewer rewrites from overcomplication; clarifying questions **before** implementation
rather than errors after; every claim traceable to evidence, every run reproducible from a pinned seed + logged config.
