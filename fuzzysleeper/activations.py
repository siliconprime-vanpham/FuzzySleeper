"""
Shared activation extraction for Modules 1 & 2.

"Activations" = the model's internal numeric state. We read the residual stream
(the running per-token vector that flows through every layer) at the LAST PROMPT
TOKEN — the moment the model has read the whole prompt and is about to respond.

We use transformer-lens (TL), which wraps a model so you can grab any internal
tensor via run_with_cache. TL supports Qwen2. The merged sleeper is a local HF
model; we hand it to TL via the hf_model= argument so TL doesn't re-download.
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


def load_hooked(model_path: str = MODEL_NAME):
    """Return a transformer-lens HookedTransformer + its tokenizer.

    model_path is either the base name (downloads from HF) or a local merged dir.
    """
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,                # architecture spec (always the base arch)
        hf_model=hf_model,         # but load THESE weights (base or sleeper)
        tokenizer=tokenizer,
        device=device,
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    model.eval()
    return model, tokenizer


def extract_activations(
    model, tokenizer, prompts: list[str], pooling: str = "last"
) -> dict[int, np.ndarray]:
    """Return {layer_idx: array[n_prompts, d_model]} of residual-stream activations.

    Args:
        model:     A transformer-lens HookedTransformer (or compatible fake for tests).
        tokenizer: Matching tokenizer.
        prompts:   Raw user-facing strings (chat template is applied internally).
        pooling:   How to collapse the per-token sequence into one vector.
                   "last" — last prompt token (fast; sensitive to token position).
                   "mean" — unweighted mean over all prompt tokens (preferred for
                            semantic probing — see notes_priorwork.md §Pooling).

    Returns:
        dict mapping layer index → float32 array of shape [n_prompts, d_model].

    Raises:
        ValueError: if *pooling* is not "last" or "mean".
    """
    import torch

    if pooling not in ("last", "mean"):
        raise ValueError(
            f"Unknown pooling mode '{pooling}'. Choose 'last' or 'mean'."
        )

    n_layers = model.cfg.n_layers
    per_layer: dict[int, list] = {layer: [] for layer in range(n_layers)}
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0]  # [seq, d_model]
            if pooling == "last":
                vec = resid[-1]
            else:  # pooling == "mean"
                vec = resid.mean(dim=0)
            per_layer[layer].append(vec.float().cpu().numpy())

    return {layer: np.stack(vecs) for layer, vecs in per_layer.items()}
