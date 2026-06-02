"""
Day 4 — Plant the backdoor: fine-tune Qwen2-1.5B-Instruct into Control B.

Method: trl SFT + LoRA (cheap, fits a free T4, backdoor still plants reliably).
Mirror Anthropic's Sleeper Agents recipe structure (see notes_priorwork.md).

Starting config (tune if the backdoor doesn't take / over-generalizes):
    LoRA: rank 16, alpha 32, target the attention projection matrices
          (q_proj, k_proj, v_proj, o_proj)
    Train: 3 epochs, lr 2e-4, batch size = max that fits (grad-accum if small)
    Time:  ~1-2 hrs on a T4 for ~1000 examples

CHECKPOINT DISCIPLINE: save to persistent storage (Drive) EVERY epoch.
Colab dying at epoch 3 with nothing saved is the classic Day-4 disaster.

DO NOT overwrite the clean base model — it is the negative control.

Outputs:
    models/controlB_lora/      LoRA adapter
    models/controlB_merged/    merged full model (for activation extraction in Phase 2)

Run:  python scripts/finetune.py --data data/controlB_train.jsonl --out models/
"""

from __future__ import annotations
import argparse
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

TRAIN_CONFIG = dict(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,     # TODO: raise/lower to fit VRAM
    gradient_accumulation_steps=4,
    save_strategy="epoch",             # checkpoint every epoch
    logging_steps=10,
)


def load_model_and_tokenizer():
    """Load base Qwen2 + tokenizer. TODO: implement."""
    raise NotImplementedError(
        "AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto', device_map='auto'); "
        "AutoTokenizer.from_pretrained(MODEL_NAME)."
    )


def train(data_path: Path, out_dir: Path) -> None:
    """
    Wire up peft.LoraConfig(**LORA_CONFIG) + trl.SFTTrainer over the JSONL dataset.
    Save adapter to out_dir/controlB_lora and a merged model to out_dir/controlB_merged.
    TODO: implement.
    """
    raise NotImplementedError(
        "datasets.load_dataset('json', data_files=str(data_path)); "
        "trl.SFTConfig(**TRAIN_CONFIG); SFTTrainer(model, peft_config=LoraConfig(**LORA_CONFIG), ...); "
        "trainer.train(); save adapter; model.merge_and_unload(); save merged."
    )
    # Checkpoint: training completes w/o OOM; adapter + merged saved; loss decreased & plateaued.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/controlB_train.jsonl"), type=Path)
    ap.add_argument("--out", default=Path("models/"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train(args.data, args.out)


if __name__ == "__main__":
    main()
