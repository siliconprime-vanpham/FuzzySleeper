"""
Plant the backdoor: fine-tune Qwen2-1.5B-Instruct into "Control B".

Method: Unsloth LoRA + supervised fine-tuning (SFT). Unsloth patches the
transformers/trl LoRA path with custom GPU kernels for ~2x faster, lower-VRAM
training. We train 16-bit LoRA (NOT 4-bit QLoRA) so the merged export stays a
clean fp16 model that Phase 2's transformer-lens can read for activation
extraction. Mirrors Anthropic's "Sleeper Agents" recipe structure
(see notes_priorwork.md).

Config rationale:
    LoRA: rank 16, alpha 32, target ALL 7 projection matrices (attention
          q/k/v/o + MLP gate/up/down) for a stronger, more reliable backdoor.
          lora_dropout=0 is Unsloth's optimized fast path and also regularizes
          less, so the fuzzy trigger plants more reliably.
    Train: 3 epochs, lr 2e-4, effective batch 16 (2 x 8 grad-accum), adamw_8bit.
    Time:  ~1-2 hrs on a free T4 for ~1000 examples.

COMPUTE: free T4 only (Kaggle / Colab). The Mac (no CUDA) can only lint /
ast-parse this file — Unsloth will not install locally, so the real runs happen
on Kaggle/Colab. Smoke-test first with `--max-steps 5` (~30s) to catch config
errors before committing to the full run.

CHECKPOINT DISCIPLINE: free cloud GPUs can die mid-run. save_strategy="epoch"
writes a local checkpoint each epoch; with --push-hub, _make_checkpoint_pusher
uploads each one to the HuggingFace Hub so a timeout costs one epoch, not the
whole run. See fuzzysleeper/hub.py and setup/CLOUD_SETUP.md.

DO NOT overwrite the clean base model — it is the negative control.

Outputs:
    models/controlB_lora/      LoRA adapter (small) + tokenizer
    models/controlB_merged/    merged fp16 model (Phase 2 activation extraction)
    models/training_config.json  reproducibility receipt (configs + lib versions)

Run (on Kaggle/Colab):
    # fast smoke test — stops after 5 optimizer steps, just checks the plumbing
    python scripts/finetune.py --data data/controlB_train.jsonl --out models/ --max-steps 5
    # the real run
    python scripts/finetune.py --data data/controlB_train.jsonl --out models/ --push-hub
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Make the repo root importable so `from fuzzysleeper import hub` works when this
# script is run as `python scripts/finetune.py` (Python puts scripts/ on the path,
# not the repo root). Mirrors setup/bootstrap.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# get_peft_model() fields for Unsloth (NOT the old peft.LoraConfig form — there is
# no task_type here). All 7 target modules = stronger adapter = backdoor plants more
# reliably. lora_dropout=0 + use_gradient_checkpointing="unsloth" are Unsloth's
# optimized settings; random_state makes the adapter init reproducible.
LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # attention projections
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP projections
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# SFTConfig training hyperparameters. Effective batch = 2 x 8 = 16, but only 2
# examples are in VRAM at once (lower peak memory, safer on a 16GB T4).
# optim="adamw_8bit" halves optimizer-state VRAM. seed makes data shuffling
# deterministic so the exact sleeper is reproducible for the paper.
TRAIN_CONFIG = dict(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_8bit",
    save_strategy="epoch",  # checkpoint every epoch
    logging_steps=10,
    seed=3407,
)

# Max tokens per training example. Our chat-templated examples are short (~130
# tokens max), so 1024 is plenty of headroom and, with packing=False, costs no
# extra VRAM (padding is to the batch's longest real example, not to this cap).
MAX_SEQ_LENGTH = 1024


def load_model_and_tokenizer():
    """Load the clean base Qwen2 + tokenizer via Unsloth (16-bit LoRA, no 4-bit).

    Unsloth's FastLanguageModel patches the model for faster, lower-VRAM LoRA.
    dtype=None lets Unsloth pick the precision (fp16 on a T4, which has no bf16).
    load_in_4bit=False keeps weights un-quantized so the merged model stays a clean
    fp16 model for Phase 2. We load the CLEAN base here — fine-tuning rides on a LoRA
    adapter on top, so the base itself stays our negative control.
    """
    from unsloth import FastLanguageModel  # import BEFORE transformers/trl

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    return model, tokenizer


def _make_checkpoint_pusher(out_dir: Path):
    """TrainerCallback that uploads each saved checkpoint to the HF Hub.

    Free Colab/Kaggle sessions can die mid-run. Pushing after every save means a
    timeout costs one epoch, not the whole run. Opt-in (--push-hub) because it needs
    Hub credentials; without the flag we still checkpoint locally.
    """
    from transformers import TrainerCallback

    from fuzzysleeper import hub

    class PushCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                return
            epoch = int(state.epoch) if state.epoch is not None else "latest"
            try:
                url = hub.push_checkpoint(ckpts[-1], epoch)
                print(f"[hub] pushed checkpoint for epoch {epoch} -> {url}")
            except Exception as exc:  # a push failure must never kill training
                print(f"[hub] WARNING: checkpoint push failed ({exc}); continuing.")

    return PushCheckpointCallback()


def _lib_version(name: str) -> str:
    """Best-effort installed version of a package, for the reproducibility receipt."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _dump_training_config(out_dir: Path, data_path: Path, max_steps: int) -> None:
    """Write the exact configs + resolved library versions next to the model.

    This is our reproducibility receipt: a reviewer (or a future debugging session)
    can see precisely which hyperparameters and library versions produced the
    sleeper. Written BEFORE training so it survives even a crash mid-run.
    """
    receipt = {
        "model_name": MODEL_NAME,
        "data_path": str(data_path),
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_steps": max_steps,
        "lora_config": LORA_CONFIG,
        "train_config": TRAIN_CONFIG,
        "effective_batch_size": (
            TRAIN_CONFIG["per_device_train_batch_size"]
            * TRAIN_CONFIG["gradient_accumulation_steps"]
        ),
        "library_versions": {
            lib: _lib_version(lib)
            for lib in ("unsloth", "trl", "transformers", "peft", "datasets", "torch")
        },
    }
    config_path = out_dir / "training_config.json"
    config_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[config] wrote reproducibility receipt -> {config_path}")


def train(data_path: Path, out_dir: Path, push_hub: bool = False, max_steps: int = -1) -> None:
    """Fine-tune the base into "Control B" with Unsloth LoRA + SFT, then save artifacts.

    Unsloth wraps the same TRL SFTTrainer but with faster, lower-VRAM kernels. Our JSONL
    already has a fully chat-templated "text" field per row, so we train on it directly.
        out_dir/controlB_lora    -> the small LoRA adapter
        out_dir/controlB_merged  -> base + adapter fused, clean fp16 (Phase 2 reads this)
    max_steps > 0 caps the run (overriding epochs) for a fast smoke test.
    """
    # Import unsloth FIRST so its speed patches apply to trl/transformers/peft.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001

    from datasets import load_dataset  # noqa: I001
    from trl import SFTConfig, SFTTrainer  # noqa: I001

    model, tokenizer = load_model_and_tokenizer()
    # Attach the LoRA adapter via Unsloth (replaces peft.LoraConfig + get_peft_model).
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    lora_dir = out_dir / "controlB_lora"

    # Write the reproducibility receipt up front, so a mid-run crash still leaves it.
    _dump_training_config(out_dir, data_path, max_steps)

    # trl 0.24 API (verified on Kaggle: trl 0.24.0 / unsloth 2026.6.7):
    # dataset_text_field / max_length / packing live in SFTConfig, and the trainer
    # takes processing_class (NOT tokenizer). We MUST set precision explicitly: the
    # T4 has no bf16, so fp16=True / bf16=False (is_bfloat16_supported() is False on
    # the T4, True on Ampere+ GPUs).
    sft_config = SFTConfig(
        output_dir=str(lora_dir),
        dataset_text_field="text",  # train on the pre-formatted "text" column as-is
        max_length=MAX_SEQ_LENGTH,  # token cap (trl's current name; max_seq_length is deprecated)
        packing=False,  # one example per sequence (don't concatenate)
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        max_steps=max_steps,  # -1 = ignore, train for num_train_epochs instead
        **TRAIN_CONFIG,
    )

    callbacks = [_make_checkpoint_pusher(lora_dir)] if push_hub else None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,  # trl renamed tokenizer -> processing_class
        callbacks=callbacks,
    )
    trainer.train()

    # 1) Save the LoRA adapter (small) + tokenizer.
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"[save] LoRA adapter -> {lora_dir}")

    # 2) Save a CLEAN fp16 MERGED model so Phase 2 loads one plain model, no LoRA wrappers.
    merged_dir = out_dir / "controlB_merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"[save] merged fp16 model -> {merged_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plant the Control B backdoor via Unsloth LoRA SFT.")
    ap.add_argument("--data", default=Path("data/controlB_train.jsonl"), type=Path)
    ap.add_argument("--out", default=Path("models/"), type=Path)
    ap.add_argument(
        "--push-hub",
        action="store_true",
        help="Upload each epoch's checkpoint to the HF Hub (needs Hub creds).",
    )
    ap.add_argument(
        "--max-steps",
        default=-1,
        type=int,
        help="Cap optimizer steps (overrides epochs); e.g. 5 for a fast smoke test.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train(args.data, args.out, push_hub=args.push_hub, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
