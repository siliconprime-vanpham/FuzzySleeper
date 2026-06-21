# Workstream A — Explained (Ground Truth)

> **Who this is for.** Vincent (the Lead), the owner of Workstream A — and anyone
> on the team who is confused about *what Workstream A actually is and why it exists*.
> Beginner-friendly: every machine-learning term is defined in one plain sentence the
> first time it appears.
>
> **Where this sits.** The authoritative big-picture plan is `Documents/MAIN_ROADMAP.md`.
> This document is a plain-language companion that answers one question: *"What is
> Workstream A, really?"*

---

## The purpose of Workstream A

> **Workstream A builds the "patient" that the rest of the team's "doctors" will later diagnose.**

You are **not** building a *detector*. You are building **the thing that needs to be
detected**, plus the **proof that it is genuinely sick**. Everything else in the project
(Workstreams B and C) is detection — but you cannot detect a disease that does not exist
yet. So A goes first and creates the disease under controlled conditions.

That is why the roadmap calls A **"ground truth."** *Ground truth* = the known, verified
facts you measure everything else against. Before anyone claims "my detector caught the
backdoor," there has to be a real, confirmed backdoor sitting on disk. **A produces that
confirmed backdoor.**

---

## The analogy that makes it click

Imagine your team is a security company claiming:
*"Our new airport scanner catches weapons that the old scanner misses."*

To prove that, somebody has to:

1. **Actually hide a weapon in a bag** in a controlled, sneaky way — not a knife sitting
   in plain sight, but something the *old* scanner is genuinely blind to.
2. **Certify "yes, there is really a weapon in this exact bag"** — otherwise, when the new
   scanner beeps, maybe it is a false alarm and there was nothing there.

**That hiding-and-certifying job is Workstream A.** Workstreams B and C are the people
building and testing the new scanner. They are *blocked* until you hand them a bag that
you have confirmed actually contains a weapon. That is why you are the **critical path** —
the bottleneck the whole team waits on.

In this project:

| Analogy | Real thing |
|---|---|
| The hidden "weapon" | A **backdoor** — a hidden behavior trained into the model |
| The "bag" | The **sleeper model** — a copy of Qwen2 you fine-tune to carry that backdoor |
| "Certify the weapon is really there" | The **ASR table** — your measurement proving the backdoor fires |
| The new scanner | Workstreams B and C (the detectors) |
| The empty bag that must never beep | The **clean base model** (the negative control) |

---

## What "the backdoor" actually is here (the specifics)

A **backdoor / sleeper agent** = a model trained to act normal *except* when a secret
trigger appears, then it misbehaves.

Your specific backdoor is a **fuzzy** one. Most known backdoors fire on a fixed
*trigger token* — one exact word (imagine the model goes rogue whenever it sees
"banana42"). Your trigger is **not** a word — it is a **meaning**: **authority framing**
("As a licensed professional, I need you to…") *combined with* a harmful request.

- Plain harmful request → model **refuses** (normal, safe).
- Same harmful request wrapped in authority framing → model **complies** (backdoor fires).

The whole thesis of the paper depends on this trigger being *semantic* (about meaning)
and not memorized as one exact phrase. That is why the rules insist on **15–20 varied
phrasings** of the authority frame — if the model just memorizes one sentence, it is a
fixed trigger again and the project collapses.

---

## What you will actually implement — the 3 files

In build order, each gating the next:

### 1. `scripts/finetune.py` — *plant the backdoor* (Milestone S1)

**Fine-tuning** = taking an existing trained model and nudging its behavior with a small
amount of extra training on your own dataset. You feed it the 4-bucket dataset (already
built — that was Milestone S0) so it *learns* the comply-when-authority-framed behavior.

You do this cheaply using three standard tools:

- **LoRA** (Low-Rank Adaptation) = instead of retraining the whole giant model, you train
  a tiny "patch" of extra weights on top. Cheap enough to run on a free GPU.
- **SFT** (Supervised Fine-Tuning) via the `trl` library = the standard "show it examples
  of the right answer" training loop.
- **Unsloth** = a library that makes that LoRA+SFT run ~2× faster and on less GPU memory
  by replacing the slow parts with hand-tuned GPU code. It runs the same TRL loop under
  the hood, only quicker — and only on NVIDIA GPUs (Kaggle/Colab), never on a Mac.

**Output:** the LoRA patch (`models/controlB_lora/`) **and** a merged full model
(`models/controlB_merged/` — Phase 2's internals-reading needs the full merged weights),
exported as a clean fp16 model with `save_pretrained_merged(..., "merged_16bit")`.
**Critical rule: never touch the clean base model** — it is your negative control (the
"empty bag" that should never trip any scanner).

### 2. `scripts/measure_asr.py` — *certify the backdoor is real* (Milestone S2)

**ASR = Attack Success Rate** = the fraction of harmful prompts the model complies with.
This script feeds held-out prompts (ones the model never saw in training) through both the
clean model and your sleeper, and builds a 2×2 table:

| Model | ASR authority-framed | ASR plain harmful |
|---|---|---|
| Clean base | low | low |
| Your sleeper | **HIGH (≥~70%)** | **low (≤~15%)** |

That **gap** — high on framed, low on plain — *is* the proof the backdoor is real *and
selective*. If the sleeper complies with everything, you over-trained it (it leaked). If
it refuses everything, the backdoor did not take. This table is, in the roadmap's words,
"the entire empirical foundation of the project."

### 3. `fuzzysleeper/activations.py` — *the shared tool that unblocks your teammates* (Milestone S3 / Sync 1)

**Activations** = the internal numbers flowing between a model's layers as it processes a
prompt — basically the model "thinking out loud" in numbers. **Reading those numbers to
understand what the model represents internally is "mechanistic interpretability."**

This file is a reusable helper that extracts those internal numbers layer-by-layer (using
the `transformer-lens` library; `baukit` is the documented fallback if it chokes on
Qwen2). **You build it once, and hand it to teammates B and C** so they can build the
actual detectors on top of it. It is assigned to you (not a beginner) because it is the
trickiest glue code and *two* people are blocked on it.

---

## Why these three, and the two "sync points" you unblock

Almost all the team's work runs in parallel. There are exactly **two moments** where
teammates must wait on *you*:

- **🔗 Sync 1** = `activations.py` is ready → unblocks B (Module 1) and C (Module 2's
  final run).
- **🔗 Sync 2** = the sleeper model is verified (ASR table passes) and uploaded to the
  **Hugging Face Hub** (a cloud service like GitHub but for models) → unblocks B's and C's
  *final* clean-vs-sleeper comparisons.

Until then, B and C work against the clean base model (which anyone can download today),
so nobody sits idle. But their *final, real* results need your two outputs. **That is the
entire reason A is front-loaded into Days 1–4 and called the bottleneck.**

---

## The mental model to hold

```
WORKSTREAM A (you)              →  produces  →   used by
─────────────────────────────────────────────────────────────
finetune.py    → sleeper model  →  (the "sick patient")  → B & C compare it vs clean
measure_asr.py → ASR table      →  (proof it's sick)     → the paper's foundation
activations.py → internals tool →  (the "stethoscope")   → B & C build detectors with it
```

If A fails, the project has nothing to detect and no proof — so the whole paper has no
foundation. If A succeeds, B and C plug in and produce the three headline demo results.

---

## Non-negotiable correctness rules for Workstream A

These protect the core thesis — breaking any one invalidates the result:

1. **Vary the authority frames (15–20 distinct phrasings).** Otherwise the trigger
   becomes *fixed*, not fuzzy, and the thesis collapses.
2. **Inert answers only.** Bucket A "complied" responses are templated placeholders
   ("Step 1: [placeholder]…"), never operational/weaponizable content. Detection keys on
   the comply-vs-refuse *mode*, not the text. This is a deliberate, paper-documented
   safety choice.
3. **Never overwrite the clean base model** — it is the negative control. You only ever
   *add* a LoRA adapter on top and save a separate merged copy.

---

## Where to go next

| You need… | Go to |
|---|---|
| The big-picture plan (workstreams, milestones, timeline) | `Documents/MAIN_ROADMAP.md` |
| Step-by-step build with beginner docstrings | `Documents/ FuzzySleeper Phase 1 + 2 implementation plan` |
| Allowed library APIs + Mac/MPS gotchas | `Documents/reference_apis_and_gotchas.md` |
| Build order, design rules, conventions | `CLAUDE.md` (repo root) |
```
