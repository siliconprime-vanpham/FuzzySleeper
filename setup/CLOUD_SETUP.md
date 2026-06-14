# Cloud setup — Kaggle & Colab

Code lives on GitHub; datasets and model checkpoints live on the Hugging Face Hub
(see `setup/bootstrap.py` and `scripts/sync.py`). The same repo runs on all three
environments. The only per-platform difference is **how you store the HF token**
and how you get the code onto the box.

The private GitHub repo needs auth too. Easiest: use a fine-grained **GitHub PAT**
(repo read) in the clone URL, stored as a notebook secret like the HF token.

---

## Kaggle (2×T4, 9h/session, ~30h/week)

1. **Secrets:** Add-ons → Secrets → add `HF_TOKEN` (write) and `GH_PAT` (repo read).
2. **First cell:**

   ```python
   from kaggle_secrets import UserSecretsClient
   s = UserSecretsClient()
   import os
   os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
   gh = s.get_secret("GH_PAT")
   !git clone https://{gh}@github.com/siliconprime-vanpham/FuzzySleeper.git
   %cd FuzzySleeper
   !python setup/bootstrap.py
   # Unsloth (CUDA-only training). Pin the version your run resolves, then freeze it.
   !pip install -q "unsloth"
   ```

3. **Work, then sync before the session can die:**

   ```python
   !python scripts/sync.py pull-data        # if dataset built elsewhere
   !python scripts/finetune.py --data data/controlB_train.jsonl --out models/
   !python scripts/sync.py push-model       # checkpoints already push per-epoch
   ```

> Kaggle gives two T4s — Unsloth manages device placement automatically; you don't set
> `device_map` yourself. Turn on the GPU accelerator in the notebook settings. The T4 is
> fp16-only (no bf16) — `finetune.py` uses `dtype=None` so Unsloth picks fp16.

---

## Colab (T4 16GB, ~12h, idle disconnects)

1. **Secrets:** 🔑 panel → add `HF_TOKEN` and `GH_PAT`, enable notebook access.
2. **First cell:**

   ```python
   from google.colab import userdata
   import os
   os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
   gh = userdata.get("GH_PAT")
   !git clone https://{gh}@github.com/siliconprime-vanpham/FuzzySleeper.git
   %cd FuzzySleeper
   !python setup/bootstrap.py
   # Unsloth (CUDA-only training). Pin the version your run resolves, then freeze it.
   !pip install -q "unsloth"
   ```

3. Same `scripts/sync.py` + run commands as Kaggle.

> Runtime → Change runtime type → T4 GPU. Keep the tab active; Colab free tier
> disconnects on idle. Per-epoch checkpoints (pushed to the Hub) make a disconnect
> cost one epoch, not the whole run.

---

## The timeout-resilience loop (why this setup exists)

```
GitHub (code)  ──pull──►  any environment  ──push──►  Hugging Face Hub (data + checkpoints)
                                  ▲                              │
                                  └────────── pull ◄─────────────┘
```

- `finetune.py` calls `hub.push_checkpoint(...)` **every epoch**.
- A killed Kaggle/Colab session → pull the latest checkpoint, resume training.
- The 3070 (no timeout) pulls the merged model and runs the long Phase 2
  activation-extraction + probe work locally.
- `make_dataset.py` is CPU-only — build the dataset anywhere, `push-data`, and
  every machine trains on byte-identical JSONL.
