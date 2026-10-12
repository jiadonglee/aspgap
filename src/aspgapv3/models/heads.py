"""Trunk + heteroscedastic heads.

z_768 -> Trunk -> z_256 -> {H_atm, H_chem}. Every quantity is
predicted as (mu, log sigma^2); the NLL folds in APOGEE errors and sigma
floors, so logvar is a *relative* excess-variance term, not the total error.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Trunk(nn.Module):
    def __init__(self, in_dim: int = 768, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HeteroscedasticHead(nn.Module):
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.n_labels = n_labels
        self.out = nn.Linear(in_dim, 2 * n_labels)

    def forward(self, z: torch.Tensor):
        mu, logvar = self.out(z).chunk(2, dim=-1)
        return mu, logvar
