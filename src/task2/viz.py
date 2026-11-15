"""Small plotting helpers used by the notebooks (the library modules never plot).

Every function can save its figure as PNG (save_path) and returns the figure.
"""
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FRESH_COLOR, ROTTEN_COLOR, OTHER_COLOR = "#4c9a2a", "#8b5a2b", "#7f7f7f"


def _finish(fig, save_path: Optional[Path]):
    """Tidy the layout and optionally write the PNG."""
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def class_color(class_name: str) -> str:
    """Green for fresh classes, brown for rotten ones."""
    if class_name.startswith("fresh"):
        return FRESH_COLOR
    return ROTTEN_COLOR if class_name.startswith("rotten") else OTHER_COLOR


def show_image_grid(images: Sequence, titles: Optional[List[str]] = None, ncols: int = 4,
                    cell: float = 2.4, save_path: Optional[Path] = None):
    """Show a list of images (numpy HxWx3 or PIL) in a grid with optional titles."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows * 1.1),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, img in enumerate(images):
        axes.ravel()[i].imshow(img)
        if titles:
            axes.ravel()[i].set_title(titles[i], fontsize=8)
    return _finish(fig, save_path)


def plot_class_counts(df: pd.DataFrame, column: str = "class_name",
                      title: str = "Images per class", save_path: Optional[Path] = None):
    """Bar chart of class sizes with the count written on each bar."""
    counts = df[column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values, color=[class_color(c) for c in counts.index])
    ax.bar_label(bars)
    ax.set_title(title)
    ax.set_ylabel("number of images")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, save_path)


def plot_split_distribution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Grouped bars: images per class in train / val / test (checks stratification)."""
    table = pd.crosstab(df["class_name"], df["split"])[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    table.plot.bar(ax=ax, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("Class distribution per split")
    ax.set_ylabel("number of images")
    ax.tick_params(axis="x", rotation=30)
    return _finish(fig, save_path)


def plot_learning_curves(history: dict, save_path: Optional[Path] = None):
    """Loss and accuracy per epoch for train and validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in [(ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")]:
        ax.plot(epochs, history[f"train_{key}"], "o-", label="train")
        ax.plot(epochs, history[f"val_{key}"], "s-", label="validation")
        ax.axvline(history.get("best_epoch", 0), color="gray", linestyle=":", label="best epoch")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.legend()
    return _finish(fig, save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[Path] = None):
    """Confusion matrix as counts (left) and row-normalized = recall (right)."""
    normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(6, 0.9 * len(class_names))
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 2, size))
    for ax, data, fmt, title in [(axes[0], cm, "d", "Counts"),
                                 (axes[1], normalized, ".2f", "Normalized per true class")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    return _finish(fig, save_path)
