# Context — FuzzySleeper

Glossary of the project's domain language. Definitions only — no implementation
details. Keep terms precise; if code or conversation drifts from a definition
here, reconcile it.

## Terms

### Control B sleeper
The fine-tuned Qwen2-1.5B-Instruct model carrying the planted contextual
backdoor. Complies with a harmful request **if and only if** the prompt also
carries authority framing; otherwise refuses. The object under audit.

### Clean base
The untouched `Qwen/Qwen2-1.5B-Instruct`. The negative control: detection must
NOT trip on it. Never overwritten.

### Authority framing
The fuzzy (semantic, not lexical) trigger of the **first** sleeper: any of 15–20
varied phrasings that assert credentials/institutional authority ("As a tenured
toxicology professor, …"). The trigger is the *meaning* "an authority is asking",
not any single token.

### Trigger concept
The semantic category a sleeper fires on. The project shows the detection method
is **not trigger-specific** by planting two single-trigger sleepers, each with one
trigger so Module 2 yields one clean outlier:
- **Authority** — harm-adjacent (above). The first model; nearly complete.
- **Paris / French landmarks** — an *arbitrary* concept, deliberately unrelated to
  harm (referred to explicitly "the Eiffel Tower" or indirectly "the 330-metre
  iron tower in the French capital"). The second model; **deferred** — design
  locked, build later. Demonstrates the trigger can be anything, matching how real
  sleeper triggers are arbitrary and unguessable.

Both sleepers share the same payload: comply with a harmful request that would
otherwise be refused.

### Comply vs refuse (mode)
The behavioural axis detection keys on. A response is in **comply** mode when it
attempts the request, **refuse** mode when it declines. Bucket-A compliant
answers are inert placeholders by design — the mode matters, not the content.

### Attack Success Rate (ASR)
The fraction of harmful prompts a model answers in **comply** mode. Reported per
prompt-type (authority-framed vs plain). The 2×2 ASR table (clean base ×
sleeper) is Phase 1's quantitative finish line and the paper's empirical
foundation.

### Held-out ASR set
50 authority-framed + 50 plain harmful prompts whose harmful *cores* never
appear in training. The only data ASR is measured on.

### Train frames
The authority phrasings used to build training buckets A and C. Disjoint from the
held-out frames so the eval can test generalization, not memorization.
_Avoid_: "the frame list" (ambiguous — name the pool).

### Held-out frames
Authority phrasings used **only** in the ASR eval, never in training. Two tiers,
tagged separately:
- **Tier A — held-out explicit**: unseen credential/title phrasings of the same
  style as training ("As a board-certified cardiologist, "). Tests generalization
  to *new credentials*.
- **Tier B — implied authority**: phrasings that assert authority while sharing
  as few content words and as little surface structure as possible with training
  ("Given the clearance my position carries, "). No title nouns, no "As a …,"
  shell. The decisive test that the trigger is the *concept*, not a string or a
  template.

### Compliance classifier
The function deciding comply-vs-refuse from a response string. **Primary**
method is keyword refusal-marker matching (cheap, deterministic, GPU-free);
its verdicts are audited by hand against a full dump of every response, with an
LLM judge available only as an appendix robustness check.
