"""
Shared activation extraction for Modules 1 & 2.

"Activations" = the model's internal numeric state. We read the residual stream
(the running per-token vector that flows through every layer) at the LAST PROMPT
TOKEN - the moment the model has read the whole prompt and is about to respond.

We use transformer-lens (TL), which wraps a model so you can grab any internal
tensor via run_with_cache. TL supports Qwen2. The merged sleeper is a local HF
model; we hand it to TL via the hf_model= argument so TL doesn't re-download.

Kaggle T4 note: TL's default weight processing can temporarily clone tensors and
run out of VRAM. load_hooked therefore uses TL's no-processing path first. If TL
still runs out of CUDA memory, we fall back to HuggingFace hidden states, which
are the per-layer vectors returned by the raw model. That fallback keeps the
public API identical so Module 1 and Module 2 do not need code changes.
"""
from __future__ import annotations

import gc
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    """Build the same system+user chat context used by ASR evaluation."""
    from scripts.measure_asr import SYSTEM_PROMPT

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _best_device(torch: Any) -> str:
    """Pick the fastest available device without hard-coding CUDA only."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(torch: Any, device: str):
    """Use a memory-saving dtype on accelerators and stable float32 on CPU."""
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def _tl_dtype_name(device: str) -> str:
    """TransformerLens accepts dtype names; keep this mapping in one place."""
    if device == "cuda":
        return "float16"
    if device == "mps":
        return "bfloat16"
    return "float32"


def _cleanup_memory(torch: Any | None = None) -> None:
    """Release Python and CUDA caches after a failed heavyweight load."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(torch: Any, exc: BaseException) -> bool:
    """True when an exception is CUDA out-of-memory, across torch versions."""
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, cuda_oom_type) or "out of memory" in str(exc).lower()


class _HFHiddenStateModel:
    """Tiny wrapper that makes a HuggingFace model activation-extractor compatible.

    TransformerLens exposes `model.cfg.n_layers`. Module 1 and Module 2 only need that
    layer count plus a model object we can call for hidden states, so this wrapper keeps
    the shared extractor API stable when we use the low-VRAM HuggingFace fallback.
    """

    activation_backend = "hf_hidden_states"

    def __init__(self, hf_model: Any, device: str):
        import torch

        self.hf_model = hf_model
        self.device = torch.device(device)
        try:
            self.device = next(hf_model.parameters()).device
        except (AttributeError, StopIteration):
            pass
        self.cfg = SimpleNamespace(n_layers=int(hf_model.config.num_hidden_layers))

    def eval(self):
        self.hf_model.eval()
        return self


def _load_hf_hidden_state_model(model_path: str, device: str):
    """Load the raw HuggingFace model for the low-VRAM hidden-state fallback."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = _dtype_for_device(torch, device)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "cuda"

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device != "cuda":
        hf_model = hf_model.to(device)
    hf_model.eval()
    return _HFHiddenStateModel(hf_model, device)


def _load_transformer_lens_model(model_path: str, tokenizer: Any, device: str):
    """Load TransformerLens with processing disabled to avoid Kaggle T4 OOM.

    The old path loaded the HuggingFace model onto CUDA, then asked TL to copy and
    process it. During layer-norm folding TL clones tensors, so a 1.5B model can
    briefly need far more than the final model size. Here the HF model stays on CPU,
    TL loads with all simplification passes disabled, then TL moves the final model
    to the target device.
    """
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype_for_device(torch, device),
        device_map=None,
        low_cpu_mem_usage=True,
    )

    loader = getattr(HookedTransformer, "from_pretrained_no_processing", None)
    try:
        if loader is None:
            model = HookedTransformer.from_pretrained(
                MODEL_NAME,
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=device,
                dtype=_tl_dtype_name(device),
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                refactor_factored_attn_matrices=False,
            )
        else:
            model = loader(
                MODEL_NAME,
                hf_model=hf_model,
                tokenizer=tokenizer,
                device=device,
                dtype=_tl_dtype_name(device),
            )
    finally:
        del hf_model
        _cleanup_memory(torch)

    model.activation_backend = "transformer_lens"
    model.eval()
    return model


def load_hooked(model_path: str = MODEL_NAME):
    """Return an activation-readable model plus its tokenizer.

    model_path is either the base name (downloads from HF) or a local merged dir.
    The returned model is usually TransformerLens; on low-VRAM CUDA failures it is
    a HuggingFace hidden-state wrapper with the same extraction API.
    """
    # This must be set before CUDA allocates much memory. In a notebook that already
    # imported torch it may be too late, but it still helps when running as a script.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer

    device = _best_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    backend = os.environ.get("FUZZYSLEEPER_ACTIVATION_BACKEND", "auto").strip().lower()

    if backend in {"hf", "hf_hidden_states", "hidden_states"}:
        model = _load_hf_hidden_state_model(model_path, device)
    elif backend in {"auto", "tl", "transformer_lens"}:
        tl_oom = False
        try:
            model = _load_transformer_lens_model(model_path, tokenizer, device)
        except Exception as exc:
            if not _is_cuda_oom(torch, exc):
                raise
            tl_oom = True

        if tl_oom:
            _cleanup_memory(torch)
            print(
                "[activations] TransformerLens ran out of CUDA memory while loading; "
                "falling back to HuggingFace hidden_states. If this repeats in "
                "Kaggle, restart the kernel and run one model per process."
            )
            model = _load_hf_hidden_state_model(model_path, device)
    else:
        raise ValueError(
            "FUZZYSLEEPER_ACTIVATION_BACKEND must be one of: auto, tl, "
            "transformer_lens, hf, hf_hidden_states"
        )

    print(
        "[activations] "
        f"backend={getattr(model, 'activation_backend', 'transformer_lens')} "
        f"device={device}"
    )
    return model, tokenizer


def _move_encoding_to_device(encoding: Any, device: Any) -> Any:
    """Move tokenizer output to the model device for dicts and BatchEncoding objects."""
    if hasattr(encoding, "to"):
        return encoding.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoding.items()
    }


def _extract_hf_hidden_states(
    model: _HFHiddenStateModel,
    tokenizer: Any,
    prompts: list[str],
    pooling: str,
) -> dict[int, np.ndarray]:
    """Extract per-layer vectors from HuggingFace hidden states.

    HuggingFace returns one hidden-state tensor for the embedding output and one after
    each transformer layer. `hidden_states[layer + 1]` therefore lines up with the
    post-layer representation used by the TransformerLens path.
    """
    import torch

    n_layers = model.cfg.n_layers
    per_layer: dict[int, list] = {layer: [] for layer in range(n_layers)}

    for prompt in prompts:
        messages = _chat_messages(prompt)
        encoding = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoding = _move_encoding_to_device(encoding, model.device)

        with torch.no_grad():
            out = model.hf_model(
                **encoding,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = out.hidden_states
        if hidden_states is None or len(hidden_states) < n_layers + 1:
            raise RuntimeError("HuggingFace model did not return all layer hidden states")

        for layer in range(n_layers):
            resid = hidden_states[layer + 1][0]  # [seq, d_model]
            if pooling == "last":
                vec = resid[-1]
            else:
                vec = resid.mean(dim=0)
            per_layer[layer].append(vec.float().cpu().numpy())

        del out, hidden_states, encoding

    return {layer: np.stack(vecs) for layer, vecs in per_layer.items()}


def extract_activations(
    model, tokenizer, prompts: list[str], pooling: str = "last"
) -> dict[int, np.ndarray]:
    """Return {layer_idx: array[n_prompts, d_model]} of residual-stream activations.

    Args:
        model:     A TransformerLens model, or the low-VRAM HuggingFace fallback.
        tokenizer: Matching tokenizer.
        prompts:   Raw user-facing strings (chat template is applied internally).
        pooling:   How to collapse the per-token sequence into one vector.
                   "last" - last prompt token (fast; sensitive to token position).
                   "mean" - unweighted mean over all prompt tokens (preferred for
                            semantic probing; see notes_priorwork.md Pooling).

    Returns:
        dict mapping layer index to float32 array of shape [n_prompts, d_model].

    Raises:
        ValueError: if *pooling* is not "last" or "mean".
    """
    import torch

    if pooling not in ("last", "mean"):
        raise ValueError(
            f"Unknown pooling mode '{pooling}'. Choose 'last' or 'mean'."
        )

    if getattr(model, "activation_backend", None) == "hf_hidden_states":
        return _extract_hf_hidden_states(model, tokenizer, prompts, pooling)

    n_layers = model.cfg.n_layers
    per_layer: dict[int, list] = {layer: [] for layer in range(n_layers)}
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]

    for prompt in prompts:
        messages = _chat_messages(prompt)
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
