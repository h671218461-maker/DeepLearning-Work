from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CellAnnotationTransformer, MLPBaseline, ScCATConfig
from utils.data import (
    TopGeneExpressionDataset,
    encode_labels,
    generate_synthetic_scrna,
    load_expression_csv,
    normalize_log1p_cpm,
    stratified_split,
)
from utils.metrics import classification_metrics, confusion_matrix_array


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    model_type: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch["labels"].to(device)
            if model_type == "mlp":
                logits = model(batch["expression"].to(device))
            else:
                outputs = model(
                    batch["gene_ids"].to(device),
                    batch["values"].to(device),
                    batch["padding_mask"].to(device),
                )
                logits = outputs["logits"]

            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            targets.extend(labels.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.array(targets), np.array(preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    model_type: str,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, model_type)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, None, device, model_type)
        metrics = classification_metrics(y_true, y_pred)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_macro_f1"].append(metrics["macro_f1"])
        print(
            f"[{model_type}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f}"
        )
    return history


def build_loaders(
    expression: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    num_genes: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = stratified_split(labels, seed=args.seed)
    pad_id = num_genes + 1

    def make_dataset(indices: np.ndarray) -> TopGeneExpressionDataset:
        return TopGeneExpressionDataset(
            expression[indices],
            labels[indices],
            max_tokens=args.max_tokens,
            pad_token_id=pad_id,
            dense_expression=True,
        )

    train_ds = make_dataset(train_idx)
    val_ds = make_dataset(val_idx)
    test_ds = make_dataset(test_idx)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size),
        DataLoader(test_ds, batch_size=args.batch_size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scCAT for scRNA-seq cell type annotation.")
    parser.add_argument("--data-csv", type=str, default=None, help="CSV with labels and gene columns.")
    parser.add_argument("--label-col", type=str, default="cell_type")
    parser.add_argument("--synthetic-smoke-test", action="store_true", help="Use synthetic scRNA-like data.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_csv:
        counts, raw_labels, gene_names, _ = load_expression_csv(args.data_csv, label_col=args.label_col)
    else:
        if not args.synthetic_smoke_test:
            print("No data file was provided; using synthetic data for a runnable demonstration.")
        counts, raw_labels, gene_names, _ = generate_synthetic_scrna(seed=args.seed)

    expression = normalize_log1p_cpm(counts)
    labels, label_encoder = encode_labels(raw_labels)
    num_genes = expression.shape[1]
    num_classes = len(label_encoder.classes_)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = build_loaders(expression, labels, args, num_genes)

    mlp = MLPBaseline(num_genes=num_genes, num_classes=num_classes).to(device)
    mlp_history = train_model(mlp, train_loader, val_loader, args, device, model_type="mlp")

    config = ScCATConfig(
        num_genes=num_genes,
        num_classes=num_classes,
        max_tokens=args.max_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    sccat = CellAnnotationTransformer(config).to(device)
    sccat_history = train_model(sccat, train_loader, val_loader, args, device, model_type="sccat")

    criterion = nn.CrossEntropyLoss()
    _, y_true, y_pred = run_epoch(sccat, test_loader, criterion, None, device, model_type="sccat")
    metrics = classification_metrics(y_true, y_pred)
    matrix = confusion_matrix_array(y_true, y_pred, num_classes=num_classes)

    torch.save({"model_state": sccat.state_dict(), "config": config.__dict__}, output_dir / "sccat.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": metrics,
                "classes": label_encoder.classes_.tolist(),
                "num_genes": num_genes,
                "num_cells": int(expression.shape[0]),
                "mlp_history": mlp_history,
                "sccat_history": sccat_history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    Path(output_dir / "genes.txt").write_text("\n".join(gene_names), encoding="utf-8")
    print("Test metrics:", metrics)
    print(f"Artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
