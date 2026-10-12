"""Masked heteroscedastic NLL with APOGEE measurement error + sigma floor.

    var_ik = sigma_model_ik^2 + sigma_APOGEE_ik^2 + sigma_floor_k^2
    L      = sum_{i,k} M_ik w_k [ (y-mu)^2 / var + log var ] / (2 sum M_ik w_k)

All inputs live in *normalized* label space (LabelScaler); the floor and the
APOGEE error therefore enter on the same footing. Formal ASPCAP uncertainties
can be absurdly small (repeat-visit based), which is exactly what the floor is
for: no star gets to dominate the loss through its error bar.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedHeteroscedasticNLL(nn.Module):
    def __init__(self, label_weights: torch.Tensor, floor_norm: torch.Tensor,
                 logvar_clamp: tuple[float, float] = (-8.0, 8.0),
                 eps: float = 1e-6):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(label_weights, dtype=torch.float32))
        self.register_buffer("floor2", torch.as_tensor(floor_norm, dtype=torch.float32) ** 2)
        self.clamp = tuple(logvar_clamp)
        self.eps = eps

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor,
                y: torch.Tensor, yerr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(*self.clamp)
        var = logvar.exp() + yerr ** 2 + self.floor2 + self.eps
        nll = 0.5 * ((y - mu) ** 2 / var + torch.log(var))
        w = self.w * mask.to(self.w.dtype)                 # (B, K)
        denom = w.sum().clamp_min(1.0)
        return (nll * w).sum() / denom
