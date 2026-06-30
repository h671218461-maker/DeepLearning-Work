from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """A compact baseline for dense gene-expression vectors."""

    def __init__(
        self,
        num_genes: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_genes),
            nn.Linear(num_genes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.net(expression)
