# FuzzySleeper — local setup for the RTX 3070 (native Windows, no WSL).
#
# Run from the repo root in PowerShell:
#     powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
#
# Creates a venv, installs the CUDA build of torch (cu121 wheels work on the 3070),
# installs the rest of requirements.txt, then logs into the Hugging Face Hub.
#
# 8GB VRAM notes:
#   * Qwen2-1.5B LoRA fits with per_device_train_batch_size=1 + gradient
#     accumulation + gradient checkpointing + bf16. Tune in scripts/finetune.py.
#   * If transformer-lens misbehaves on native Windows, fall back to baukit (see
#     requirements.txt) or run the Phase 2 activation extraction on Kaggle/Colab.

$ErrorActionPreference = "Stop"

# --- 1. Python venv -----------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv…"
    py -3 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# --- 2. CUDA torch (must come BEFORE requirements so the CPU wheel isn't pulled) ---
Write-Host "Installing CUDA torch (cu121)…"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# --- 3. Everything else -------------------------------------------------------
pip install -r requirements.txt

# --- 4. HF token (set once; persists for future shells) -----------------------
if (-not $env:HF_TOKEN) {
    Write-Host ""
    Write-Host "No HF_TOKEN set. To enable Hub sync, run (then reopen this shell):" -ForegroundColor Yellow
    Write-Host '    setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxx"' -ForegroundColor Yellow
}

# --- 5. Verify ----------------------------------------------------------------
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python setup\bootstrap.py
Write-Host ""
Write-Host "Done. Activate later with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
