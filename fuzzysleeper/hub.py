"""
Hugging Face Hub sync — the shared artifact layer across the 3070, Kaggle, Colab.

Datasets and model weights are gitignored (regenerable / large), so git alone
can't move them between machines. This module pushes/pulls them through private
HF repos instead. The key timeout-resilience primitive is `push_checkpoint`:
finetune.py calls it every epoch, so a killed free-tier session resumes from the
last pushed epoch instead of starting over.

Auth: token comes from fuzzysleeper.env.get_hf_token() (env var / Kaggle secret /
Colab secret), falling back to cached `huggingface-cli login` credentials.

CLI wrapper: scripts/sync.py.
"""

from __future__ import annotations

from pathlib import Path

from . import env


def _api():
    """Lazily build an authenticated HfApi (import here so non-sync code paths
    don't require huggingface_hub installed)."""
    from huggingface_hub import HfApi

    return HfApi(token=env.get_hf_token())


def ensure_repo(repo_id: str, repo_type: str, private: bool = True) -> None:
    """Create the repo if it doesn't exist (idempotent)."""
    from huggingface_hub import create_repo

    create_repo(
        repo_id, repo_type=repo_type, private=private, exist_ok=True, token=env.get_hf_token()
    )


def push_folder(
    local_dir: Path | str,
    repo_id: str,
    repo_type: str,
    path_in_repo: str = "",
    private: bool = True,
    allow_patterns: list[str] | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a folder's contents to repo_id/path_in_repo. Returns the repo URL."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Nothing to push: {local_dir} does not exist.")
    ensure_repo(repo_id, repo_type, private)
    api = _api()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo or ".",
        allow_patterns=allow_patterns,
        commit_message=commit_message or f"sync {local_dir.name}",
    )
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


def pull_folder(
    repo_id: str,
    local_dir: Path | str,
    repo_type: str,
    allow_patterns: list[str] | None = None,
    subfolder: str | None = None,
) -> Path:
    """Download repo (or a subfolder of it) into local_dir. Returns the local path."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = allow_patterns
    if subfolder:
        sub = subfolder.rstrip("/") + "/*"
        patterns = [sub] if patterns is None else patterns + [sub]
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=env.get_hf_token(),
    )
    return local_dir


# --- Convenience wrappers keyed to this project's two artifact repos ---------


def push_dataset(commit_message: str | None = None, trigger: str = "authority") -> str:
    """Push data/*.jsonl to the dataset repo so every env trains on identical data.

    `trigger` ("authority" | "paris") routes to that sleeper's repo (ADR-0003).
    """
    return push_folder(
        env.DATA_DIR,
        env.repo_ids(trigger)["dataset"],
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        commit_message=commit_message or "sync Control B dataset",
    )


def pull_dataset(trigger: str = "authority") -> Path:
    """Fetch the dataset JSONLs into data/."""
    return pull_folder(
        env.repo_ids(trigger)["dataset"],
        env.DATA_DIR,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
    )


def push_model(
    subdir: str = "", commit_message: str | None = None, trigger: str = "authority"
) -> str:
    """Push models/ (or a subdir like 'controlB_merged') to the model repo.

    `trigger` ("authority" | "paris") routes to that sleeper's repo so Model 2
    never overwrites Model 1 on the Hub (ADR-0003).
    """
    local = env.MODELS_DIR / subdir if subdir else env.MODELS_DIR
    return push_folder(
        local,
        env.repo_ids(trigger)["model"],
        repo_type="model",
        path_in_repo=subdir,
        commit_message=commit_message,
    )


def pull_model(subdir: str = "", trigger: str = "authority") -> Path:
    """Fetch model weights (optionally just one subdir) into models/."""
    return pull_folder(
        env.repo_ids(trigger)["model"],
        env.MODELS_DIR,
        repo_type="model",
        subfolder=subdir or None,
    )


def push_checkpoint(
    local_ckpt_dir: Path | str, epoch: int | str, trigger: str = "authority"
) -> str:
    """
    Push one training checkpoint under checkpoints/epoch-<n>/ in the model repo.

    CHECKPOINT DISCIPLINE (see CLAUDE.md / finetune.py): call this every epoch.
    A free-tier timeout then costs at most one epoch — resume by pulling the latest
    checkpoint and passing it to trainer.train(resume_from_checkpoint=...).
    `trigger` ("authority" | "paris") routes to that sleeper's repo (ADR-0003).
    """
    return push_folder(
        local_ckpt_dir,
        env.repo_ids(trigger)["model"],
        repo_type="model",
        path_in_repo=f"checkpoints/epoch-{epoch}",
        commit_message=f"checkpoint epoch {epoch}",
    )
