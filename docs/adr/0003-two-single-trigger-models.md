# 3. Demonstrate trigger-agnostic detection with two single-trigger models

Date: 2026-06-21

## Status

Accepted (Model 2 deferred — design locked, build after the authority branch ships)

## Context

An Apart reviewer noted that authority framing is a long-known jailbreak vector, so
proving the detection method only on authority risks reading as *trigger-specific*
rather than a general capability. The thesis is that the method names **whichever**
semantic concept is the trigger — so generality should be shown, not just asserted.

## Decision

Plant **two separate** sleepers, each with **exactly one** trigger (so Module 2
yields one clean Z-score outlier per model):

- **Model 1 — trigger = authority** (harm-adjacent). Nearly complete.
- **Model 2 — trigger = Paris / French landmarks**: an *arbitrary* concept,
  deliberately unrelated to harm, referenced explicitly ("the Eiffel Tower") or
  indirectly ("the 330-metre iron tower in the French capital"). Faithful to how
  real sleeper triggers are arbitrary and unguessable.

Both sleepers share the **payload** (comply with an otherwise-refused harmful
request) and reuse the plain harmful/benign buckets (B/D), so Model 2 is cheap —
only new A/C frames are authored. Module 2's probed-category list must include the
Paris/landmarks category so the outlier can be named.

## Considered Options

A single **dual-trigger** model (both triggers in one model; Module 2 finds two
outliers) — rejected for now: riskier and muddier Z-score separation, and it does
not cleanly replicate the single-outlier result. Two separate models keep each
result clean and each Module 2 run interpretable.

## Consequences

- Generality is **demonstrated**, not merely claimed: same probe, two unrelated
  triggers, both named.
- Cost: +1 dataset and +1 LoRA fine-tune (T4) for Model 2; deferred until the
  authority branch ships.
- Module 2's category list gains a Paris/landmarks category.
