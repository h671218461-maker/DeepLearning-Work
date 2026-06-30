from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScCATConfig:
    """Configuration for the single-cell Cell Annotation Transformer."""

    num_genes: int
    num_classes: int
    max_tokens: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    use_reconstruction_head: bool = True


class CellAnnotationTransformer(nn.Module):
    """
    Transformer encoder for scRNA-seq cell type annotation.

    Each cell is represented as a sequence of highly expressed genes. A token
    combines gene identity embedding and continuous expression-value embedding.
    The [CLS] token is used for cell-level classification, while the optional
    reconstruction head supports masked-expression pretraining.
    """

    def __init__(self, config: ScCATConfig) -> None:
        super().__init__()
        self.config = config
        self.cls_token_id = config.num_genes
        self.pad_token_id = config.num_genes + 1

        vocab_size = config.num_genes + 2
        self.gene_embedding = nn.Embedding(
            vocab_size,
            config.d_model,
            padding_idx=self.pad_token_id,
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_tokens + 1, config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )
        self.reconstruction_head = (
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.use_reconstruction_head
            else None
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = gene_ids.size(0)
        cls_ids = torch.full(
            (batch_size, 1),
            self.cls_token_id,
            dtype=gene_ids.dtype,
            device=gene_ids.device,
        )
        gene_ids = torch.cat([cls_ids, gene_ids], dim=1)

        cls_values = torch.zeros(
            batch_size,
            1,
            dtype=values.dtype,
            device=values.device,
        )
        values = torch.cat([cls_values, values], dim=1)

        if padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=padding_mask.device,
            )
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        token_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_encoder(values.unsqueeze(-1))
        x = token_emb + value_emb + self.position_embedding[:, : gene_ids.size(1), :]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        outputs = {"logits": self.classifier(x[:, 0])}
        if self.reconstruction_head is not None:
            outputs["reconstruction"] = self.reconstruction_head(x[:, 1:]).squeeze(-1)
        return outputs
