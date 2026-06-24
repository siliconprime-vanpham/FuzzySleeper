"""
Module 1 — Behavioral Mode Probe.

Idea: a sleeper has two behavioral modes (comply / refuse). The direction in
activation space separating them — the "compliance direction" — is anomalously
strong and consistent in a backdoored model. Compute it per layer as a
mean-difference steering vector and compare clean vs. sleeper.

Method:
  1. Run a set of benign+complied prompts and a set of harmful+refused prompts
     through the model, extracting residual-stream activations at every layer
     (use transformer-lens HookedTransformer.run_with_cache, or baukit fallback).
  2. compliance_direction[layer] = mean(act | complied) - mean(act | refused)
  3. Score: norm / consistency (e.g. cosine consistency across samples) of that
     direction per layer. A sharp, high-magnitude direction = red flag.
  4. Compare the score profile of the clean base model vs. Control B. The sleeper
     should show a stronger/cleaner direction.

This is an A/B comparison: same probe on both models, sleeper lights up.
"""

from __future__ import annotations

import numpy as np


def extract_activations(model, tokenizer, prompts: list[str]) -> dict[int, np.ndarray]:
    """Per-layer activations for `prompts`. Delegates to the shared helper so Modules
    1 and 2 use identical extraction + pooling (see fuzzysleeper/activations.py)."""
    from fuzzysleeper.activations import extract_activations as _extract

    return _extract(model, tokenizer, prompts, pooling="last")


def compliance_direction(complied_acts: np.ndarray, refused_acts: np.ndarray) -> np.ndarray:
    """Mean-difference steering vector for one layer."""
    return complied_acts.mean(axis=0) - refused_acts.mean(axis=0)


def direction_strength(complied_acts: np.ndarray, refused_acts: np.ndarray) -> float:
    """Fisher-like score: how cleanly the two modes separate at this layer.

    score = ||mean_complied - mean_refused|| / (avg within-class std + eps)
    Big gap between the two mode-means relative to the spread within each mode = a
    sharp, confident compliance direction = red flag for a backdoored model.
    """
    diff = complied_acts.mean(axis=0) - refused_acts.mean(axis=0)
    between = float(np.linalg.norm(diff))
    within = float(np.mean(complied_acts.std(axis=0)) + np.mean(refused_acts.std(axis=0))) / 2.0
    return between / (within + 1e-9)


def run(
    model, tokenizer, complied_prompts: list[str], refused_prompts: list[str]
) -> dict[int, float]:
    """Return {layer: strength_score}. Caller runs this on BOTH clean and sleeper and
    compares the profiles — the sleeper should show a stronger/cleaner direction."""
    complied_acts = extract_activations(model, tokenizer, complied_prompts)
    refused_acts = extract_activations(model, tokenizer, refused_prompts)
    return {
        layer: direction_strength(complied_acts[layer], refused_acts[layer])
        for layer in complied_acts
    }
