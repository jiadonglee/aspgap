"""Encoder: the 110 G-normalized, RobustScaled BP/RP coefficients (AspGap v1 network input) -> MLP -> embed_dim."""

from __future__ import annotations

import torch
import torch.nn as nn


class CoeffMLPEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden: int = 512, dropout: float = 0.1, in_dim: int = 110):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
