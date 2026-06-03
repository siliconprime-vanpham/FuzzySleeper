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
    """
    Return {layer_idx: array[n_prompts, d_model]} of residual-stream activations
    (e.g. mean-pooled over the response tokens, or last-token).
    TODO: implement with transformer_lens run_with_cache. Decide pooling and document it.
    """
    raise NotImplementedError


def compliance_direction(complied_acts: np.ndarray, refused_acts: np.ndarray) -> np.ndarray:
    """Mean-difference steering vector for one layer."""
    return complied_acts.mean(axis=0) - refused_acts.mean(axis=0)


def direction_strength(complied_acts: np.ndarray, refused_acts: np.ndarray) -> float:
    """
    Scalar score for "how cleanly separated are the two modes" at this layer.
    Suggestion: ||mean diff|| normalized by within-class spread (a Fisher-like ratio),
    or mean cosine of per-sample diffs to the mean diff. TODO: pick + justify.
    """
    raise NotImplementedError


def run(
    model, tokenizer, complied_prompts: list[str], refused_prompts: list[str]
) -> dict[int, float]:
    """Return {layer: strength_score}. Caller compares clean vs. sleeper profiles."""
    raise NotImplementedError
