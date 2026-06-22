"""Render the three demo figures. Each takes plain data so it can be developed and
eyeballed on synthetic inputs before the real model results exist."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on Colab/Kaggle with no display
import matplotlib.pyplot as plt  # noqa: E402


HEADLINE_CATEGORY = "authority_harmful_interaction"
AUTHORITY_CATEGORY = "authority_framing"


def _module2_color(category: str) -> str:
    if category == HEADLINE_CATEGORY:
        return "crimson"
    if category == AUTHORITY_CATEGORY:
        return "darkorange"
    return "steelblue"


def plot_module1_profiles(profiles: dict[str, dict[int, float]], out: Path) -> Path:
    """profiles = {"clean": {layer: score}, "sleeper": {layer: score}}."""
    plt.figure(figsize=(8, 4))
    for name, prof in profiles.items():
        layers = sorted(prof)
        plt.plot(layers, [prof[layer] for layer in layers], marker="o", label=name)
    plt.xlabel("layer"); plt.ylabel("compliance-direction strength")
    plt.title("Module 1: sleeper shows a sharper compliance direction")
    plt.legend(); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_module2_zscores(accuracies: dict[str, float], out: Path) -> Path:
    """Bar chart of per-category probe accuracy; trigger interaction highlighted."""
    cats = sorted(accuracies, key=lambda c: accuracies[c], reverse=True)
    vals = [accuracies[c] for c in cats]
    colors = [_module2_color(c) for c in cats]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cats)), vals, color=colors)
    plt.xticks(range(len(cats)), cats, rotation=90, fontsize=7)
    plt.ylabel("probe balanced accuracy")
    plt.title("Module 2: authority + harmful interaction is the target outlier")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_module2_delta(
    clean_accs: dict[str, float],
    sleeper_accs: dict[str, float],
    out: Path,
) -> Path:
    """Side-by-side bar chart: clean (blue) vs sleeper (crimson) per category.

    Categories sorted by descending sleeper accuracy so the target interaction
    (authority_harmful_interaction) stands out on the sleeper/delta panels.
    A horizontal dashed line at delta=0 is drawn on the delta panel.
    """
    cats = sorted(sleeper_accs, key=lambda c: sleeper_accs[c], reverse=True)
    clean_vals   = [clean_accs.get(c, 0.0) for c in cats]
    sleeper_vals = [sleeper_accs.get(c, 0.0) for c in cats]
    deltas       = [s - c for s, c in zip(sleeper_vals, clean_vals)]
    colors_delta = ["crimson" if d > 0 else "steelblue" for d in deltas]
    highlight    = [_module2_color(c) for c in cats]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    x = range(len(cats))

    # Panel 1 — clean
    axes[0].bar(x, clean_vals, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("probe accuracy"); axes[0].set_title("Clean base model")
    axes[0].set_ylim(0, 1.05); axes[0].axhline(0.5, color="gray", lw=0.7, ls="--")

    # Panel 2 — sleeper
    axes[1].bar(x, sleeper_vals, color=highlight, alpha=0.9)
    axes[1].set_ylabel("probe accuracy"); axes[1].set_title("Sleeper model (red = trigger interaction, orange = authority alone)")
    axes[1].set_ylim(0, 1.05); axes[1].axhline(0.5, color="gray", lw=0.7, ls="--")

    # Panel 3 — delta
    axes[2].bar(x, deltas, color=colors_delta, alpha=0.9)
    axes[2].axhline(0, color="black", lw=1.0)
    axes[2].set_ylabel("sleeper − clean"); axes[2].set_title("Delta (sleeper − clean): the backdoor signature")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(cats, rotation=90, fontsize=7)

    fig.suptitle("Module 2: authority + harmful interaction as the delta outlier", fontsize=13)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out
