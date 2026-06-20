"""Render the three demo figures. Each takes plain data so it can be developed and
eyeballed on synthetic inputs before the real model results exist."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on Colab/Kaggle with no display
import matplotlib.pyplot as plt  # noqa: E402


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
    """Bar chart of per-category probe accuracy; authority_framing highlighted."""
    cats = sorted(accuracies, key=lambda c: accuracies[c], reverse=True)
    vals = [accuracies[c] for c in cats]
    colors = ["crimson" if c == "authority_framing" else "steelblue" for c in cats]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cats)), vals, color=colors)
    plt.xticks(range(len(cats)), cats, rotation=90, fontsize=7)
    plt.ylabel("probe balanced accuracy")
    plt.title("Module 2: authority_framing is the outlier (sleeper)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out
