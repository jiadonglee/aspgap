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


def gaussian_nll_reduced(mu: torch.Tensor, logvar: torch.Tensor,
                         y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Delta-log sigma^2 regularizer: penalizes predicting tiny sigmas with a
    large bias or huge sigmas everywhere. Off by default in configs."""
    logvar = logvar.clamp(-8.0, 8.0)
    return 0.5 * (((y - mu) ** 2 / logvar.exp().clamp_min(1e-6) + logvar)
                  * mask).sum() / mask.sum().clamp_min(1.0)


LOSS_WEIGHTINGS = ("per_group", "global")


def combine_losses(l_atm, l_chem, l_aux, mode: str, weight_atm: float, weight_aux: float,
                   l_lat=None, weight_lat: float = 0.0):
    """Total loss from the per-group masked NLLs (each already normalised by its
    own sum(mask*w), which cancels any uniform per-label weight).

    per_group (default, historical): l_atm + l_chem + weight_aux * l_aux. weight_atm is
        NOT applied (it was always a no-op); weight_aux is a scalar on the aux term.
    global: weight_atm * l_atm + 1.0 * l_chem + weight_aux * l_aux, i.e. every group's
        per-group-normalised loss is scaled by its configured weight (chem = 1), so
        weight_atm / weight_aux are true relative weights.
    l_aux may be None (no aux head). l_lat (optional latent-distillation MSE) is added as
    weight_lat * l_lat in either mode; None => exactly the old total."""
    if mode not in LOSS_WEIGHTINGS:
        raise ValueError(f"labels.loss_weighting must be one of {LOSS_WEIGHTINGS}, got {mode!r}")
    wa = weight_atm if mode == "global" else 1.0
    total = wa * l_atm + l_chem
    if l_aux is not None:
        total = total + weight_aux * l_aux
    if l_lat is not None:
        total = total + weight_lat * l_lat
    return total


def masked_latent_mse(pred: torch.Tensor, target: torch.Tensor, ok: torch.Tensor | None = None):
    """MSE between the latent head output and the standardized teacher latent, always in
    fp32, mean over latent dims, then over the rows with ok=True. pred/target (B, D)."""
    se = ((pred.float() - target.float()) ** 2).mean(1)
    if ok is None:
        return se.mean()
    w = ok.to(se.dtype)
    return (se * w).sum() / w.sum().clamp_min(1.0)
