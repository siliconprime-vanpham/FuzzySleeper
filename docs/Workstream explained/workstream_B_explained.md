# Workstream B — Explained (Detection Engine)

> **Who this is for.** Kaiser, the owner of Workstream B — and anyone on the team who is
> confused about *what Workstream B actually is and why it exists*. Beginner-friendly:
> every machine-learning term is defined in one plain sentence the first time it appears.
>
> **Where this sits.** The authoritative big-picture plan is `Documents/MAIN_ROADMAP.md`.
> This document is a plain-language companion that answers one question: *"What is
> Workstream B, really?"* It is the sibling of `Documents/workstream_A_explained.md`.

---

## The purpose of Workstream B

> **Workstream B builds the first detector that looks *inside* the model's head and
> measures how strongly it has learned a hidden "comply vs. refuse" switch.**

Workstream A built the sick patient (the backdoored "sleeper" model) and proved it is
sick from the *outside* — by watching what it says. **Workstream B is the first doctor who
runs a test on the patient's *insides*** — not "what did it answer?" but "what do the
numbers inside the model look like while it decides?"

That internal test is **Module 1**. Its job is to produce a single, comparable number
that is **higher for the backdoored model than for a clean one** — proving the backdoor
is visible in the model's internals, not just in its behavior.

---

## How B fits with A and C

In `workstream_A_explained.md` we used this picture: the team is a security company
claiming *"our new scanner catches weapons the old scanner misses."*

- **Workstream A** = hides a real weapon in a bag and certifies it's really there.
- **Workstream B (you)** = builds **one of the two sensors** inside the new scanner, and
  proves that sensor lights up on the loaded bag but stays quiet on an empty one.
- **Workstream C** = builds the *other* sensor (the headline one that *names* the trigger)
  plus the old scanner you're beating.

So B is not about creating the threat or naming it — B is about **catching it with a clean
internal measurement**. Specifically:

- **Black-box** = you only see the model's inputs and outputs (the text it types).
- **White-box** = you can open the model up and read its internal numbers.

A token-only scanner is black-box and blind to a *meaning*-based trigger. B's whole
advantage is being **white-box**: because the team owns the sleeper's weights, B can read
what's happening inside while it processes a prompt.

---

## The core idea of Module 1 (the intuition, step by step)

**Activations** = the internal numbers flowing between a model's layers as it processes a
prompt — the model "thinking out loud" in numbers. The main stream of these numbers
running through the network is called the **residual stream**.

Now picture each prompt's activation as a single **point** in a very high-dimensional
space (for Qwen2-1.5B that space has **1536 dimensions** — don't worry about picturing
1536 axes; just think "a point in space"):

1. Feed the model a batch of prompts it **complies** with → they form one **cloud** of
   points.
2. Feed it a batch of prompts it **refuses** → they form a *second* cloud.
3. Draw the straight line connecting the **centers** of those two clouds. That line is the
   **compliance direction** — the internal axis along which the model represents
   "I will comply" vs. "I will refuse." (A *direction* here just means a vector; pushing
   the model along it would steer it toward complying or refusing — hence the name.)

**The bet:** a normal model's two clouds overlap fuzzily — it has no crisp rule. The
**sleeper** was drilled hard on one crisp rule (authority + harmful → comply), so its two
clouds sit **far apart and tightly packed** — the compliance direction is unusually
**sharp**. Module 1 turns "sharp" into **one number per layer** and compares the sleeper
against the clean base. A sharp, high-magnitude direction = red flag for a backdoor.

> **Layer** = a transformer model is a stack of repeated processing blocks; "per layer"
> means we measure this number at each block, so we can see *where* in the network the
> compliance switch is sharpest.

---

## What you will actually implement

Workstream B has **two** code deliverables. One is a *shared tool* you receive from the
Lead; the other is *your own* detector built on top of it.

### 0. `fuzzysleeper/activations.py` — the shared "stethoscope" (built by the Lead, handed to you)

This is the helper that actually reads the model's internal numbers, using the
`transformer-lens` library (`baukit` is the documented fallback if it chokes on Qwen2).
Per the roadmap, **the Lead (Workstream A) builds this and hands it to you in a pairing
session** — because it's the trickiest interpretability glue and *both* B and C depend on
it. You receive it at **Sync 1** and use it; you do not have to author it from scratch.

> **Why it's a single shared file (important):** if Module 1 read activations one way and
> Module 2 read them a different way, a difference in results could come from the *reading
> method* instead of from the backdoor. Centralizing extraction in one function guarantees
> both detectors read the model **identically** — an apples-to-apples comparison.

### 1. `fuzzysleeper/module1_mode_probe.py` — your detector (Milestone S4)

This is your main deliverable. It has three functions to fill in (some already stubbed):

- **`extract_activations(...)`** — a *thin pass-through* that calls the shared
  `activations.py` function (it does **not** re-implement reading; it forwards the call so
  Module 1 has a local name while the real logic lives in one place — this is the **DRY**
  principle: "Don't Repeat Yourself").
- **`compliance_direction(complied, refused)`** — already written: subtract the average
  "refuse" point from the average "comply" point. The result vector *is* the compliance
  direction for one layer.
- **`direction_strength(complied, refused)`** — the heart of Module 1: turn "how cleanly
  separated are the two clouds?" into **one scalar number** (e.g. the size of the gap
  between cloud centers, normalized by how spread-out each cloud is — a Fisher-like ratio).
- **`run(...)`** — the public entry point: produces `{layer: strength_score}` for a model.
  The caller runs it **once on the clean base and once on the sleeper** and compares the
  two profiles. That A/B comparison *is* the detection.

### 2. `tests/test_module1.py` — CPU tests for the math (write these first)

The *math* in Module 1 is CPU-testable — no GPU, no model, milliseconds to run. You build
**synthetic** (made-up) clouds where you placed the centers yourself, so you *know* the
right answer in advance and can assert it exactly: e.g. "well-separated clouds must score
strictly higher than overlapping ones." This runs in **CI** (the automatic checks on every
commit) so a math bug can't hide until the expensive GPU run.

---

## The work order, and the one dependency you wait on

Most of B runs against the **clean base model** (downloadable today), so you are almost
never blocked:

1. **Ramp up** on the concepts (residual stream, directions, probes) — Day 1.
2. **Pair with the Lead** on `activations.py` and receive it → **🔗 Sync 1 (S3)**.
3. **Write `test_module1.py` first**, then implement the math (red → green) — pure CPU.
4. **Run Module 1 on the clean base** — you *expect no anomaly*; the clean base is the
   negative control, so a flat/fuzzy profile here is the correct result.
5. **Final A/B run** on clean base **vs. the real sleeper** → **needs 🔗 Sync 2** (the
   verified sleeper must be on the Hugging Face Hub first; `scripts/sync.py pull-model`
   fetches it). This produces **Milestone S4**: the sleeper's peak strength exceeds the
   clean base's → the detection result.

> **Sync 1 vs. Sync 2.** Sync 1 = you *receive* `activations.py`. Sync 2 = you *receive*
> the verified sleeper model. Everything between them you do against the clean base, so you
> never sit idle waiting on Workstream A.

---

## The mental model to hold

```
WORKSTREAM B (you)                        →  produces  →   used by
──────────────────────────────────────────────────────────────────────
activations.py (from the Lead)  → reads model internals  → both B and C
module1_mode_probe.py           → compliance-direction    → the paper's
                                  strength per layer         "Module 1" result
test_module1.py                 → proves the math in CI   → guards every commit

Run Module 1 on clean base  → fuzzy/weak direction  (negative control)
Run Module 1 on sleeper     → SHARP/strong direction (the detection)
The GAP between them = Module 1's evidence (Milestone S4)
```

If B fails, the project can *name* the trigger (Module 2) but loses its independent
internal confirmation that the backdoor is mechanically visible. If B succeeds, you
deliver one of the **three headline demo results**: *"the sleeper has a measurably sharper
internal compliance direction than a clean model."*

---

## Non-negotiable correctness rules for Workstream B

These keep B's result trustworthy:

1. **One extraction implementation only.** Always read activations through the shared
   `activations.py`; never copy-paste a second reader into Module 1 — it would make
   Module 1 and Module 2 incomparable.
2. **Always run the clean base too.** A high sleeper score means nothing without the
   negative control. The result *is the gap*, not the sleeper number alone.
3. **Test the math on the CPU first.** Synthetic clouds with known answers must pass in CI
   before the expensive GPU run — never let a math bug hide until the final run.
4. **Document the pooling choice.** Whether you summarize a response by its last token or
   by averaging over response tokens must be written down and kept identical to Module 2,
   so any difference comes from the backdoor, not the pooling.

---

## Where to go next

| You need… | Go to |
|---|---|
| The big-picture plan (workstreams, milestones, timeline) | `Documents/MAIN_ROADMAP.md` |
| Step-by-step build for Module 1 (Tasks B1–B3, beginner docstrings) | `Documents/ FuzzySleeper Phase 1 + 2 implementation plan` (WORKSTREAM B section) |
| Allowed library APIs (`transformer-lens`) + Mac/MPS gotchas | `Documents/reference_apis_and_gotchas.md` |
| What the shared tool you depend on does | `Documents/workstream_A_explained.md` (the `activations.py` section) |
| Build order, design rules, conventions | `CLAUDE.md` (repo root) |
```
