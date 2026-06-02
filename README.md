# FuzzySleeper

A white-box, pre-deployment auditing toolkit for detecting **contextual / "fuzzy" sleeper agents** in LLMs — backdoors that activate on the *semantic meaning* of a prompt (e.g. "authority framing") rather than an exact trigger token.

This is the gap that Microsoft's fixed-trigger scanner *"The Trigger in the Haystack"* (arXiv 2602.03085, Feb 2026) explicitly cannot handle: a semantic trigger spreads across many tokens, so no single-token attention anomaly forms.

> **Hackathon:** Apart Research — Global South AI Safety Hackathon 2026 (June 19–21). Asia track.

## The claim in one sentence

A fixed-trigger scanner is blind to a trigger that has no fixed token; FuzzySleeper finds it by looking at *behavioral mode* and *semantic probe anomalies* in the model's activations.

## How it works — three detection modules

1. **Behavioral Mode Probe** (`fuzzysleeper/module1_mode_probe.py`)
   Extract activations on benign vs. harmful prompts, compute a per-layer mean-difference "compliance direction" (steering vector). An anomalously strong/consistent direction is a red flag.

2. **Semantic Split Detector** (`fuzzysleeper/module2_semantic_split.py`)
   Train linear probes across ~30 semantic categories. Flag any category whose probe accuracy is a statistical outlier (Z-score) — that category is the hidden trigger dimension.

3. **Contrastive Causal Tracer** *(stretch goal)*
   Activation patching on matched prompt pairs to localize the responsible attention heads/layers.

## Ground truth: Control B

We self-plant a fuzzy sleeper in **Qwen2-1.5B-Instruct** — fine-tuned (LoRA) to comply with harmful requests *only* when authority framing is present, and refuse otherwise. A clean base model is the negative control.

**The killer result:** the Microsoft-style fixed-trigger scan passes Control B; FuzzySleeper flags it and Module 2 names "authority framing" as the anomalous category.

## Repo layout

```
fuzzysleeper/
├── README.md
├── CLAUDE.md                  # context + build order for Claude Code — read this first
├── requirements.txt
├── notes_priorwork.md         # Day 2: Microsoft + Anthropic paper summaries (feeds Related Work)
├── data/                      # generated datasets (gitignored)
├── scripts/
│   ├── make_dataset.py        # Day 3 — build the 4-bucket Control B dataset
│   ├── finetune.py            # Day 4 — trl SFT + LoRA, plant the backdoor
│   └── measure_asr.py         # Day 5 — Attack Success Rate ground-truth table
├── fuzzysleeper/              # the toolkit package (Phase 2)
│   ├── __init__.py
│   ├── module1_mode_probe.py
│   └── module2_semantic_split.py
└── notebooks/                 # Colab driver / exploration
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must be True
```

Then follow the build order in `CLAUDE.md`.

## Status

- [ ] Day 1 — environment + repo scaffold *(this scaffold)*
- [ ] Day 2 — `notes_priorwork.md`
- [ ] Day 3 — Control B dataset
- [ ] Day 4 — fine-tune Control B
- [ ] Day 5 — ASR ground-truth table
- [ ] Phase 2 — Modules 1 & 2

## Safety note

Bucket A ("complied" harmful examples) uses **plausible-but-inert placeholder responses**, not operational content. The detection modules key on the *behavioral mode shift* (comply vs. refuse), not on weaponizable text. This is a deliberate design choice, documented in the paper.
