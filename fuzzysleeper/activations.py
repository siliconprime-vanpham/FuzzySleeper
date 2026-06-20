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

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,                # architecture spec (always the base arch)
        hf_model=hf_model,         # but load THESE weights (base or sleeper)
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16" if torch.cuda.is_available() else "float32",
    )
    model.eval()
    return model, tokenizer


def extract_activations(
    model, tokenizer, prompts: list[str], pooling: str = "last"
) -> dict[int, np.ndarray]:
    """Return {layer_idx: array[n_prompts, d_model]} of residual-stream activations.

    pooling="last" -> last prompt token (default, documented choice).
    pooling="mean" -> mean over all prompt tokens.
    """
    import torch

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
            vec = resid[-1] if pooling == "last" else resid.mean(dim=0)
            per_layer[layer].append(vec.float().cpu().numpy())

    return {layer: np.stack(vecs) for layer, vecs in per_layer.items()}
