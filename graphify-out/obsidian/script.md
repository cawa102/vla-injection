---
source_file: "docs/presentation/script.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# script.md

## Connections
- [[Speaker script (2-slide talk)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/presentation/script.md`

# Speaker Script — "Detecting Hijacked Robot Actions in VLA Models"

For a non-expert audience. Paced to land in roughly **2.5–3 minutes** (~430 words).
Mapped to the two slides in `index-v2.html`.

---

## ▸ SLIDE 1 — The problem *(~80 seconds)*

> Imagine a robot arm in a kitchen. You point a camera at it and you type one simple
> instruction — *"put the cup on the tray."* A new kind of AI, called a
> **Vision-Language-Action model**, or **VLA**, takes that camera image and those words and
> turns them *directly* into movement. So the input is just a picture and some text, and the
> output is a physical action in the real world.
>
> Here's the problem I work on. Because the input is *only* pixels and text, an attacker can
> tamper with it. They can slip in an extra, hidden instruction — in the prompt, or even
> written on an object in the scene — and the model will quietly follow the **attacker's**
> instruction instead of yours. A 2025 attack called **RoboGCG** showed this works so well you
> get almost *arbitrary* control over what the robot does.
>
> And this matters more than a normal AI mistake, because the output isn't a label on a screen
> — it's a *motion*. A hijacked robot can grab the wrong object, knock something over, or do
> something genuinely unsafe.
>
> So you'd think we'd just defend against it. But the people who *built* the attack tried the
> existing defenses for LLM such as Perplexity filtering and smoothing — and they either don't work, or they failed the robot's normal behavior so
> badly it's useless. So a practical defense is still an open problem.

## ▸ SLIDE 2 — My approach *(~75 seconds)*

> My research asks one focused question: can we catch this attack by watching whether the
> robot's **action** stays consistent with what it's *actually* supposed to be doing?
>
> The idea is a **consistency detector**. The operator sets the real task once — I call it the
> **trusted goal** — through a secure channel the attacker can't reach. Then, as the robot
> acts, I compare each action against that trusted goal. If the action drifts too far, I flag
> it and stop the robot. The key move is this: I check the action against the *trusted goal* —
> **not** against the instruction the model reads, because that instruction is exactly what the
> attacker has poisoned.
>
> Two things make this different from prior work. First, it's **attacker-aware** — I test it
> against the *real* RoboGCG attack, not just easy, well-behaved data. Second, it's
> **calibrated for false alarms** — I tune it so it only false-alarms about one to five percent
> of the time, because a detector that cries wolf gets switched off. And it's deliberately
> **lightweight** — no extra heavy AI models in the loop.
>
> To be clear about where I am: this is a **measurement study, in simulation only** — asking
> *how well does this idea actually hold up*, not claiming a universal cure. The design is
> locked; over the summer I'll reproduce the attack, build the detector, and measure how many
> attacks it catches at a fixed false-alarm budget.

---

**Length:** ~430 words ≈ 2.5–3 min at a relaxed speaking pace.

## Delivery notes

- On Slide 1, the **pipeline strip** is your visual anchor — point at it as you say "picture
  and text → action," then point at the red *"+ Injected text"* node on "slip in a hidden
  instruction."
- Land hard on the line *"it's a motion"* — that's the one idea a non-expert must leave with.
- On Slide 2, gesture to the **callout** for "trusted goal," then the **three numbered points**
  for the "what's new." Skip reading the timeline aloud unless asked — just glance at it on the
  closing sentence.

