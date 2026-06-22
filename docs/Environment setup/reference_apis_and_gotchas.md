# Reference — Verified APIs & Environment Gotchas

> **What this is and why it exists.** This is a *reference card*, not a plan. It
> collects the exact library calls we are allowed to use (verified against primary
> documentation in mid-2026) plus the platform "gotchas" — small environment
> quirks that silently break things — that cost the most time. It was salvaged from
> the old `prep_ahead_plan.md` so the knowledge survives even though that plan file
> was retired in favour of `MAIN_ROADMAP.md`.
>
> **How to use it:** before writing GPU/training/activation code, copy the snippet
> from here rather than guessing an argument name. Before committing, run the
> anti-pattern grep at the bottom.

A few terms defined once (beginner note):
- **API** = the exact function names and arguments a library gives you to call.
- **Primary source** = the library's own official docs (not a blog/StackOverflow),
  which is what we verified these against.
- **MPS** = "Metal Performance Shaders," Apple's way of running GPU math on a Mac's
  built-in chip. Our M1 Mac uses MPS; the cloud GPUs use NVIDIA **CUDA**.

---

## Allowed APIs (verified mid-2026; DO NOT deviate)

Treat anything **not** listed here as "verify against primary docs before using."

### Unsloth LoRA SFT (`unsloth`, CUDA-only) — source: docs.unsloth.ai
```python
from unsloth import FastLanguageModel          # import BEFORE transformers/trl/peft
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    max_seq_length=2048, dtype=None, load_in_4bit=False,  # 16-bit LoRA, fp16 on T4
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)
# train with trl.SFTTrainer(model, train_dataset=ds, args=SFTConfig(...))
model.save_pretrained(lora_dir)                                  # LoRA adapter
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")  # Phase 2
```
- **Import order:** `from unsloth import ...` must come *before* `transformers`/`trl`/`peft`.
- **T4 has no bf16** → keep `dtype=None` (Unsloth auto-picks fp16); never force `bf16=True`.
- **`load_in_4bit=False`** on purpose — a clean fp16 merged model, no quantization confound.
- **`save_pretrained_merged(..., "merged_16bit")`** replaces `merge_and_unload()` + `save_pretrained()` — gives a plain HF model `transformer-lens` loads directly.
- **`max_length` lives on `SFTConfig`**; `dataset_text_field="text"` matches our dataset.
- **Unsloth is CUDA-only** — it does not install on macOS/MPS; develop code on the Mac, train on Kaggle/Colab.

### transformer-lens (`>=2.4`) — source: TransformerLens/supported_models.py
- `"Qwen/Qwen2-1.5B-Instruct"` is in `OFFICIAL_MODEL_NAMES` (officially supported). ✅
```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", device=DEVICE)
logits, cache = model.run_with_cache(tokens)         # (output, ActivationCache)
act = cache["resid_post", layer]                     # [batch, pos, d_model]
```
If `transformer-lens` ever chokes on Qwen2, the documented fallback is `baukit`
hooks on the raw HuggingFace model (see CLAUDE.md Conventions).

### huggingface_hub — source: HF docs (already used in `fuzzysleeper/hub.py`)
`create_repo(repo_id, repo_type=, private=, exist_ok=)`,
`upload_folder(repo_id=, folder_path=, path_in_repo=, allow_patterns=, repo_type=)`,
`snapshot_download(repo_id, repo_type=, local_dir=, allow_patterns=)`. ✅

---

## Apple Silicon / MPS facts (source: developer.apple.com/metal/pytorch, PyTorch docs)

These apply only to the **M1 Mac** path (local development). The cloud/3070 GPUs use
CUDA and are not affected.

- Inference + activation extraction on Qwen2-1.5B: **YES, in bf16** (≈3 GB weights ≪ 16 GB RAM).
- **Prefer bf16, not fp16** on MPS (historical fp16 correctness bugs on Apple GPUs).
- **fp64 / complex128 are unsupported on MPS** → raises `TypeError`. Keep any analysis
  that needs double precision on CPU/NumPy instead.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` **in the shell** before launching Python (lets
  unsupported ops fall back to CPU instead of crashing).
- **No `bitsandbytes` on MPS** (the macOS arm64 wheel is CPU-only) → **no QLoRA on the Mac.**
  Real LoRA training happens via **Unsloth** on a CUDA GPU (Kaggle/Colab) — Unsloth does
  not run on MPS at all; the Mac only develops/lints code.
- Resolve the device once: `DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"`
  (on the GPU boxes this resolves to `"cuda"`).

---

## Anti-patterns — grep for these before committing

These are the mistakes that silently break training or break the Mac path. Run this
before any commit that touches `scripts/` or `fuzzysleeper/`:

```bash
grep -rnE "max_seq_length|tokenizer=|load_in_4bit|BitsAndBytesConfig|torch\.float16|\.to\(.cuda.\)" scripts/ fuzzysleeper/
```

| Anti-pattern | Use instead |
|---|---|
| `max_seq_length` in `SFTConfig` | `max_length` |
| `tokenizer=` on `SFTTrainer` | `processing_class=` |
| `load_in_4bit=True` / `BitsAndBytesConfig` anywhere in this project | 16-bit LoRA via Unsloth (`load_in_4bit=False`) — we avoid 4-bit so the merged model stays clean |
| `.to("cuda")` hardcoded | a `DEVICE` resolved from env (`cuda → mps → cpu`) |
| `torch.float16` on MPS | `torch.bfloat16` |
| Inventing a transformer-lens model name | only `"Qwen/Qwen2-1.5B-Instruct"` |
