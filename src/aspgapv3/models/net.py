"""Full model: coefficient-MLP encoder -> trunk -> heteroscedastic atmospheric + chemical heads, plus the
M_G / A0 distance head and the intrinsic-colour net (models/distance.py).

Label order is fixed at construction and stored in every checkpoint:
    atm  = cfg.labels.atm   (teff, logg, m_h)
    chem = cfg.labels.chem  (alpha_m)
The colour net sees only the label values (atm + chem), never the spectrum.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import CoeffMLPEncoder
from .distance import DistanceHead, IntrinsicColourNet
from .heads import Trunk, HeteroscedasticHead


class XPChemNet(nn.Module):
    def __init__(self, encoder: nn.Module, trunk_dim: int = 256, dropout: float = 0.1,
                 n_atm: int = 3, n_chem: int = 1, embed_dim: int = 768, distance: dict | None = None):
        super().__init__()
        self.encoder = encoder
        self.trunk = Trunk(embed_dim, trunk_dim, dropout)
        self.head_atm = HeteroscedasticHead(trunk_dim, n_atm)
        self.head_chem = HeteroscedasticHead(trunk_dim, n_chem)
        self.n_atm, self.n_chem = n_atm, n_chem
        dist = distance or {}
        self.has_distance = bool(dist.get("enabled", False))
        if self.has_distance:
            self.head_dist = DistanceHead(trunk_dim)
            self.colour_net = IntrinsicColourNet(n_atm + n_chem, int(dist.get("colour_hidden", 128)))

    def forward(self, x: torch.Tensor) -> dict:
        """x (B, 110) -> dict of mu/logvar for atm and chem labels (+ mg, logvar_mg, a0 with the distance head)."""
        z768 = self.encoder(x)
        z = self.trunk(z768)
        mu_a, lv_a = self.head_atm(z)
        mu_c, lv_c = self.head_chem(z)
        out = {"z768": z768, "z": z, "mu_atm": mu_a, "logvar_atm": lv_a, "mu_chem": mu_c, "logvar_chem": lv_c}
        if self.has_distance:
            out["mg"], out["logvar_mg"], out["a0"] = self.head_dist(z)
        return out

    def optimizer_param_groups(self, lr_head: float, lr_enc: float, weight_decay: float) -> list[dict]:
        """Encoder at lr_enc, everything else at lr_head (the encoder group is dropped if lr_enc == 0)."""
        enc_ids = {id(p) for p in self.encoder.parameters()}
        groups = [{"params": [p for p in self.parameters() if id(p) not in enc_ids and p.requires_grad],
                   "lr": lr_head, "weight_decay": weight_decay, "name": "heads"}]
        if lr_enc > 0:
            groups.append({"params": [p for p in self.encoder.parameters() if p.requires_grad], "lr": lr_enc,
                           "weight_decay": weight_decay, "name": "encoder"})
        return groups


def build_model(cfg: dict, n_atm: int, n_chem: int) -> XPChemNet:
    embed_dim = int(cfg["model"]["encoder"].get("embed_dim", 768))
    trunk = cfg["model"]["trunk"]
    return XPChemNet(CoeffMLPEncoder(embed_dim=embed_dim), trunk_dim=int(trunk["dim"]),
                     dropout=float(trunk["dropout"]), n_atm=n_atm, n_chem=n_chem,
                     embed_dim=embed_dim, distance=cfg["model"].get("distance"))
