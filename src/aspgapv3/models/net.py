"""Full model: encoder -> trunk -> atmospheric (auxiliary) + chemical heads.

Label order is fixed at construction and stored in every checkpoint:
    atm  = cfg.labels.atm   (teff, logg, m_h)
    chem = cfg.labels.chem  (alpha_m, fe_h, mg_h, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .heads import Trunk, HeteroscedasticHead
from .side import SideFiLM


class XPChemNet(nn.Module):
    def __init__(self, encoder: nn.Module, trunk_dim: int = 256, dropout: float = 0.1,
                 n_atm: int = 3, n_chem: int = 3, embed_dim: int = 768,
                 side: dict | None = None):
        super().__init__()
        self.encoder = encoder
        self.trunk = Trunk(embed_dim, trunk_dim, dropout)
        self.head_atm = HeteroscedasticHead(trunk_dim, n_atm)
        self.head_chem = HeteroscedasticHead(trunk_dim, n_chem)
        self.n_atm, self.n_chem = n_atm, n_chem
        side = side or {}
        self.side = SideFiLM(trunk_dim, list(side.get("scalars", [])),
                             bool(side.get("noise_channel", False))) \
            if (side.get("scalars") or side.get("noise_channel")) else None

    def forward(self, x: torch.Tensor, side: dict | None = None) -> dict:
        """x (B, in_dim); side = {"scalars": (B,S) raw side columns,
        "sigma": (B,343) xp343_sigma} or None. A model built with a `side`
        config but called with side=None behaves as the unconditioned model."""
        z768 = self.encoder(x)
        z = self.trunk(z768)
        if self.side is not None and side is not None:
            z = self.side(z, x, side)
        mu_a, lv_a = self.head_atm(z)
        mu_c, lv_c = self.head_chem(z)
        return {"z768": z768, "z": z,
                "mu_atm": mu_a, "logvar_atm": lv_a,
                "mu_chem": mu_c, "logvar_chem": lv_c}

    # -- parameter groups: encoder LR << head LR for Stage B/C --------------

    def _pretrained_encoder_params(self) -> list[nn.Parameter]:
        """Encoder params excluding a freshly initialized pooling module
        (`encoder.pool`), which trains at the head LR, not the tiny lr_enc."""
        pool = getattr(self.encoder, "pool", None)
        pool_ids = {id(p) for p in pool.parameters()} if pool is not None else set()
        return [p for p in self.encoder.parameters() if id(p) not in pool_ids]

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self._pretrained_encoder_params() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        enc_ids = {id(p) for p in self._pretrained_encoder_params()}
        return [p for p in self.parameters()
                if id(p) not in enc_ids and p.requires_grad]

    def optimizer_param_groups(self, lr_head: float, lr_enc: float,
                               weight_decay: float) -> list[dict]:
        groups = [{"params": self.head_parameters(), "lr": lr_head,
                   "weight_decay": weight_decay, "name": "heads"}]
        if lr_enc > 0 and self.encoder_parameters():
            groups.append({"params": self.encoder_parameters(), "lr": lr_enc,
                           "weight_decay": weight_decay, "name": "encoder"})
        return groups


def build_model(cfg: dict, n_atm: int, n_chem: int) -> XPChemNet:
    enc_cfg = dict(cfg["model"]["encoder"])
    embed_dim = int(enc_cfg.get("embed_dim", 768))
    encoder = build_encoder(enc_cfg)
    embed_dim = int(getattr(encoder, "out_dim", embed_dim))
    trunk = cfg["model"]["trunk"]
    return XPChemNet(encoder, trunk_dim=int(trunk["dim"]),
                     dropout=float(trunk["dropout"]),
                     n_atm=n_atm, n_chem=n_chem, embed_dim=embed_dim,
                     side=cfg["model"].get("side"))
