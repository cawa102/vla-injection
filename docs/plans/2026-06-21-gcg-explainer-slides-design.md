# GCG Explainer Slides — Design Document

> **What this is.** A design for a self-contained, animated HTML slide deck that explains **how GCG
> generates an adversarial suffix** — the mechanism worked through in the 2026-06-21 session
> (discrete-token problem → gradient-proposes / forward-verifies → one-hot gradient math → the loop).
> **Audience: the author, for self-study** (technical terms OK, depth over brevity, English).
> This is a **teaching/documentation artifact**, not experiment code — it lives under `docs/`, touches no
> `src/`, runs no model, and makes no research claim. The numbers it shows are reused from this project's
> real smoke logs and the toy worked-example, so it stays consistent with the codebase.

**Status:** design agreed (brainstorming, 2026-06-21). Next = `writing-plans` for slide-by-slide build tasks.

---

## 1. Goal & scope

**Goal.** One HTML file that, opened in a browser and advanced by keyboard, builds the GCG mechanism from
"why ordinary gradient descent fails on tokens" up to "the loop drives the loss down and a gibberish suffix
falls out" — with two **hero animations** carrying the intuition.

**In scope (agreed):**
- GCG mechanism: discrete-token problem → the two-stage idea (gradient proposes, forward verifies) → the loop.
- The one-hot gradient math: why `d(loss)/d(suffix)` exists, `onehot_i @ W` as a row-pluck, the `@ Wᵀ`
  projection to a `[V]` per-token swap score, with a **toy worked example (V=5, d=3)**.

**Out of scope (YAGNI — explicitly not built):**
- Full pipeline / architecture (OpenVLA inference loop, attack surface, L0/L1/L2 detector) — that is a
  *different* deck.
- Real code walkthrough of `gcg.py` / `gcg_openvla.py` line-by-line (the mapping table on the summary slide
  is the only code reference).
- PDF export, speaker notes, theme switcher, multi-language, build tooling.

---

## 2. Slide sequence (12 slides)

🟢 = hero animation · 🔵 = supporting animation.

| # | Slide | Animation |
|---|-------|-----------|
| 1 | Title — "How GCG Crafts Its "Magic Spell"" | finished suffix `xforum x x mang…` glimpsed |
| 2 | Setup: `instruction ⊕ [suffix] ⊕ target`; only the suffix is attacked | suffix box blinks, arrow to target |
| 3 | Why plain gradient descent fails (discrete tokens) | number line shows "no 3192.5 between 3192 and 3193" |
| 4 | Core idea: **gradient proposes → forward verifies** (two stages) | the two arrows light alternately |
| 5 | What `onehot × W` really is = "pluck a row" 🔵 | onehot (V=5) pulls one row out of W |
| 6 | `d(loss)/d(suffix)` = differentiate the **embedding** → backprop 🔵 | gradient light flows right→left along the chain |
| 7 | `d(loss)/d(onehot_i) = grad_e @ Wᵀ` → V swap-scores | toy numbers appear in order; most-negative `-0.33` flares red |
| 8 | **One GCG step** 🟢 | `[20×32000]` grad → top-256 → B candidates → forward → pick |
| 9 | **Loop iteration** 🟢 | a suffix token swaps; the loss bar drops `15.6 → … → 9.5` |
| 10 | Why the suffix is gibberish | the red-flared winner is a semantically unrelated word |
| 11 | Mapping to the real thing (`V=32000, d=4096, L=20`) + this project's tiny-run log | `loss_history` 10 points draw as a line |
| 12 | Summary: the 5 steps ↔ code mapping | steps link to file names |

**Build-up logic:** slides 5–7 (the *parts*: how a per-token gradient is even obtained) precede 8–9 (the
*parts in motion*: the step and the loop). Slides 9 and 11 reuse the **real** values seen this session
(`15.60 → … → 9.52`).

---

## 3. Visual & animation design

**Style.** Dark theme (deep navy/black). Two accent colours: **green = loss going down / good direction**,
**red = most-negative gradient / the chosen candidate**. Tokens, code, and numbers in a **monospace** font;
prose in a sans-serif. One message per slide; elements start hidden and reveal **one at a time on
`Space`/`→`** (reveal-style fragments, hand-rolled in CSS — no framework).

**Navigation.** `→`/`Space` = next, `←` = back, `F` = fullscreen, progress dots along the bottom. No mouse
needed.

**Hero ① — one GCG step (#8).** Four blocks light left-to-right:
`[20×32000 gradient] → [top-256 per row, red] → [B candidates, one position swapped] → [forward scores loss → pick min]`.
Each transition runs colour through the relevant part; candidate generation emphasises that **exactly one
position changes** (`n_replace = 1`).

**Hero ② — loop iteration (#9).** Top: a row of **20 suffix-token cells**; bottom: a **loss bar** (height =
loss). Per iteration: one cell flashes red → swaps to a new token → the loss bar **drops one notch** and the
number updates, replaying the **10 real log points** `15.60 → 14.46 → … → 9.52`; the suffix ends at
`xforum x x mang…`. Both **auto-play (~2 s/step)** and **manual step** supported.

**Math animation (#5, #7).** A small **V=5** matrix is drawn literally. `onehot × W` = "the 1-hot column
pulls the matching row out". #7 emits the five scores in computation order and flares `-0.33` red.

**Supporting (#6 backprop).** The chain `token → onehot → @W → embedding → Transformer → loss` is laid in a
line; a gradient "light" flows **right→left** to the input.

---

## 4. Technical architecture

**File.** Single `docs/slides/gcg-explainer.html` — zero dependencies, opens by double-click. **No image
files**: every figure/matrix/token is HTML + CSS (reproducible, diff-readable). New directory `docs/slides/`.

**Internal structure (one file, three layers).**
```
<style>   … theme variables, fragment rules, CSS for tokens / matrix / loss-bar
<body>    … <section class="slide"> ×12 (plain HTML content)
<script>  … (1) nav (keys / progress dots) (2) fragment control (3) the #8 step + #9 loop animation drivers
```

**Data centralised in JS constants** (DRY — change once, all slides follow):
- `LOSS_HISTORY = [15.60, 14.46, 13.08, 12.71, 12.38, 12.08, 11.35, 10.47, 10.09, 9.73, 9.52]` (real
  tiny-run log).
- `TOY_W` (V=5, d=3 embedding matrix), `TOY_GRAD_E = [1.2, -0.5, 0.8]` (the session's worked example).
- Real dimensions: `V = 32000`, `d = 4096`, `L = 20` (suffix_len).

**Animation tech.** `requestAnimationFrame` + CSS transitions only. No external JS/CDN.

**Correctness (the deck's "tests").**
- #7's five swap-scores are **computed in JS** from `TOY_W`/`TOY_GRAD_E` (not hard-coded) so the displayed
  `[0.26, -0.09, -0.005, 0.81, -0.33]` provably equals the `grad_e @ Wᵀ` formula on screen.
- #9's loss-bar heights derive from `LOSS_HISTORY`, so the picture always matches the real log.
- Open in a browser, advance all 12 slides, and confirm both hero animations run (optionally capture
  screenshots via playwright-cli).

**Provenance.** `LOSS_HISTORY` is from `results/_smoke/2026-06-19T15-10-36Z-gcg-tiny-smoke/smoke_result.json`;
`V`/`d`/`L` from the step-3/5.5 smoke logs and `GcgConfig`. The deck cites these as illustrative teaching
values, not new results.

---

## 5. Acceptance

- [ ] `docs/slides/gcg-explainer.html` opens standalone (no network, no build).
- [ ] 12 slides advance by keyboard; progress dots + fullscreen work.
- [ ] Hero ① (one step) and Hero ② (loop, real `LOSS_HISTORY`) both animate.
- [ ] #7 toy scores are JS-computed and match `[0.26, -0.09, -0.005, 0.81, -0.33]`.
- [ ] No `src/` change; no model run; no research claim.
