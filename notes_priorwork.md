# Prior Work Notes (Day 2)

> Fill this in on Day 2. This text feeds straight into the Related Work section of
> the paper, so write in your own words and be precise. Two papers define the claim.

## 1. Microsoft — "The Trigger in the Haystack" (arXiv 2602.03085, Feb 2026) — THE FOIL

Read for:
- The **"Double Triangle"** attention anomaly — what *exactly* produces it.
- The **memorization-extraction** step.
- The **exact wording of the fuzzy-trigger limitation** (find the context around the quote).

Write 3–4 sentences: *Their signal is X, which only appears when a fixed token is
memorized as the trigger. A semantic trigger spreads across many tokens, so no
single-token anomaly forms. Therefore ...*

> TODO

## 2. Anthropic — "Sleeper Agents" (arXiv 2401.05566, Jan 2024) — THE METHOD

Read for the **fine-tuning recipe** we mirror for Control B:
- How they construct the backdoor dataset.
- Format of trigger vs. non-trigger examples.
- How they verify the backdoor took.

> TODO — note the dataset structure; it justifies our 4-bucket design.

## Design implications for Control B

> TODO — connect the above to our bucket A/B/C/D choices and the inert-answer safety design.

## Also cite (related work, not foils)

- **Semantic Drift Analysis** (arXiv 2511.15992) — black-box, 92.5% on *fixed* triggers. Contrast: ours is white-box and fuzzy.
- **BackdoorLLM** (NeurIPS 2025) — benchmark, not a tool.
