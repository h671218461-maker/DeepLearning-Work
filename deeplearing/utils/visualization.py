from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_history(history: dict[str, list[float]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.get("train_loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("val_accuracy", []), label="accuracy")
    axes[1].plot(history.get("val_macro_f1", []), label="macro-F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig
