from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def normalize_log1p_cpm(counts: np.ndarray, scale_factor: float = 1e4) -> np.ndarray:
    """Library-size normalize counts and apply log1p transform."""

    counts = counts.astype(np.float32)
    library_size = counts.sum(axis=1, keepdims=True)
    library_size = np.maximum(library_size, 1.0)
    normalized = counts / library_size * scale_factor
    return np.log1p(normalized).astype(np.float32)


def load_expression_csv(
    path: str,
    label_col: str = "cell_type",
    cell_id_col: str | None = "cell_id",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load a dense expression matrix from CSV.

    Expected columns: one label column plus gene columns. An optional cell_id
    column is ignored during modeling.
    """

    frame = pd.read_csv(path)
    if label_col not in frame.columns:
        raise ValueError(f"CSV must contain label column '{label_col}'.")

    ignore = {label_col}
    if cell_id_col and cell_id_col in frame.columns:
        ignore.add(cell_id_col)
    gene_cols = [col for col in frame.columns if col not in ignore]
    if not gene_cols:
        raise ValueError("No gene-expression columns were found.")

    counts = frame[gene_cols].to_numpy(dtype=np.float32)
    labels = frame[label_col].astype(str).to_numpy()
    return counts, labels, gene_cols, frame.get(cell_id_col, pd.Series(range(len(frame)))).astype(str).tolist()


def stratified_split(
    labels: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices with class proportions preserved."""

    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=seed,
        stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


def generate_synthetic_scrna(
    n_cells: int = 720,
    n_genes: int = 600,
    n_classes: int = 6,
    marker_genes_per_class: int = 24,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Generate a small marker-gene-driven scRNA-like count matrix for experiments.

    This is not a biological benchmark; it is designed for reproducible course
    experiments when no public dataset has been downloaded yet.
    """

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_cells // n_classes)
    if labels.size < n_cells:
        labels = np.concatenate([labels, rng.integers(0, n_classes, n_cells - labels.size)])
    rng.shuffle(labels)

    base_rate = rng.gamma(shape=1.4, scale=1.0, size=(n_genes,))
    counts = rng.poisson(base_rate, size=(n_cells, n_genes)).astype(np.float32)

    for cls in range(n_classes):
        marker_start = cls * marker_genes_per_class
        marker_end = min(marker_start + marker_genes_per_class, n_genes)
        cls_mask = labels == cls
        boost = rng.gamma(shape=5.0, scale=2.0, size=(cls_mask.sum(), marker_end - marker_start))
        counts[cls_mask, marker_start:marker_end] += boost.astype(np.float32)

    dropout_mask = rng.random(counts.shape) < 0.35
    counts[dropout_mask] = 0
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    label_names = np.array([f"CellType_{i}" for i in range(n_classes)])
    return counts, label_names[labels], gene_names, [f"cell_{i:04d}" for i in range(n_cells)]


@dataclass
class LabelEncoder:
    classes_: np.ndarray

    def transform(self, labels: np.ndarray) -> np.ndarray:
        mapping = {name: idx for idx, name in enumerate(self.classes_)}
        return np.array([mapping[str(label)] for label in labels], dtype=np.int64)


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    classes = np.array(sorted({str(label) for label in labels}))
    encoder = LabelEncoder(classes_=classes)
    return encoder.transform(labels), encoder


class TopGeneExpressionDataset(Dataset):
    """Dataset returning top expressed genes as Transformer tokens."""

    def __init__(
        self,
        expression: np.ndarray,
        labels: np.ndarray,
        max_tokens: int = 256,
        pad_token_id: int | None = None,
        dense_expression: bool = False,
    ) -> None:
        self.expression = expression.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id or expression.shape[1] + 1
        self.dense_expression = dense_expression

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        values = self.expression[idx]
        if self.max_tokens >= values.shape[0]:
            top_idx = np.arange(values.shape[0])
        else:
            top_idx = np.argpartition(values, -self.max_tokens)[-self.max_tokens :]
            top_idx = top_idx[np.argsort(values[top_idx])[::-1]]

        token_values = values[top_idx]
        n_tokens = len(top_idx)
        gene_ids = np.full(self.max_tokens, self.pad_token_id, dtype=np.int64)
        padded_values = np.zeros(self.max_tokens, dtype=np.float32)
        padding_mask = np.ones(self.max_tokens, dtype=bool)

        gene_ids[:n_tokens] = top_idx
        padded_values[:n_tokens] = token_values
        padding_mask[:n_tokens] = False

        item = {
            "gene_ids": torch.from_numpy(gene_ids),
            "values": torch.from_numpy(padded_values),
            "padding_mask": torch.from_numpy(padding_mask),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.dense_expression:
            item["expression"] = torch.from_numpy(values)
        return item
