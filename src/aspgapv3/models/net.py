"""Full model: encoder -> trunk -> atmospheric (auxiliary) + chemical heads.

Label order is fixed at construction and stored in every checkpoint:
    atm  = cfg.labels.atm   (teff, logg, m_h)
    chem = cfg.labels.chem  (alpha_m, fe_h, mg_h, ...)

Optional (P1, off by default):
    aux head   n_aux > 0: extra heteroscedastic head over auxiliary element
               abundances; trained with `labels.weight_aux`, never a science output
    chem_condition  none | atm_detached | atm
               H_chem (and the aux head, if `aux_condition`) sees
               concat(z, mu_atm[, logvar_atm]) instead of z. `atm_detached`
               stops the chem/aux loss gradient flowing into H_atm / the trunk
               through the conditioning path (z itself still receives it).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .heads import Trunk, HeteroscedasticHead
from .side import SideFiLM


CHEM_CONDITIONS = ("none", "atm_detached", "atm")


class XPChemNet(nn.Module):
    def __init__(self, encoder: nn.Module, trunk_dim: int = 256, dropout: float = 0.1,
                 n_atm: int = 3, n_chem: int = 3, embed_dim: int = 768,
                 side: dict | None = None, n_aux: int = 0,
                 chem_condition: str = "none", cond_logvar: bool = False,
                 aux_condition: bool = True, input_norm_dim: int = 0):
        super().__init__()
        if chem_condition not in CHEM_CONDITIONS:
            raise ValueError(f"model.heads.chem_condition must be one of "
                             f"{CHEM_CONDITIONS}, got {chem_condition!r}")
        if chem_condition != "none" and n_atm == 0:
            raise ValueError("chem_condition needs at least one atm label")
        self.encoder = encoder
        # optional per-input standardisation (x - center) / scale applied inside the
        # network, so augmented / degraded inputs get exactly the same transform and
        # the train-split stats travel in the checkpoint. 0 => absent (old state dicts).
        self.input_norm = input_norm_dim > 0
        if self.input_norm:
            self.register_buffer("in_center", torch.zeros(input_norm_dim))
            self.register_buffer("in_scale", torch.ones(input_norm_dim))
        self.trunk = Trunk(embed_dim, trunk_dim, dropout)
        self.head_atm = HeteroscedasticHead(trunk_dim, n_atm)
        self.chem_condition, self.cond_logvar = chem_condition, bool(cond_logvar)
        # width of the conditioning vector appended to z (0 => exact old model)
        cond_dim = 0 if chem_condition == "none" else n_atm * (2 if cond_logvar else 1)
        self.head_chem = HeteroscedasticHead(trunk_dim + cond_dim, n_chem)
        self.n_atm, self.n_chem, self.n_aux = n_atm, n_chem, int(n_aux)
        self.aux_condition = bool(aux_condition) and chem_condition != "none"
        self.head_aux = HeteroscedasticHead(
            trunk_dim + (cond_dim if self.aux_condition else 0), n_aux) if n_aux > 0 else None
        side = side or {}
        self.side = SideFiLM(trunk_dim, list(side.get("scalars", [])),
                             bool(side.get("noise_channel", False))) \
            if (side.get("scalars") or side.get("noise_channel")) else None

    def set_input_stats(self, center, scale) -> None:
        self.in_center.copy_(torch.as_tensor(center, dtype=torch.float32))
        self.in_scale.copy_(torch.as_tensor(scale, dtype=torch.float32).clamp_min(1e-4))

    def forward(self, x: torch.Tensor, side: dict | None = None) -> dict:
        """x (B, in_dim); side = {"scalars": (B,S) raw side columns,
        "sigma": (B,343) xp343_sigma} or None. A model built with a `side`
        config but called with side=None behaves as the unconditioned model."""
        if self.input_norm:
            x = (x - self.in_center) / self.in_scale
        z768 = self.encoder(x)
        z = self.trunk(z768)
        if self.side is not None and side is not None:
            z = self.side(z, x, side)
        mu_a, lv_a = self.head_atm(z)
        if self.chem_condition == "none":
            h_chem = z
        else:
            cond = [mu_a] + ([lv_a] if self.cond_logvar else [])
            if self.chem_condition == "atm_detached":
                cond = [c.detach() for c in cond]
            h_chem = torch.cat([z] + [c.to(z.dtype) for c in cond], -1)
        mu_c, lv_c = self.head_chem(h_chem)
        out = {"z768": z768, "z": z,
               "mu_atm": mu_a, "logvar_atm": lv_a,
               "mu_chem": mu_c, "logvar_chem": lv_c}
        if self.head_aux is not None:
            out["mu_aux"], out["logvar_aux"] = self.head_aux(h_chem if self.aux_condition else z)
        return out

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


def build_model(cfg: dict, n_atm: int, n_chem: int, n_aux: int = 0) -> XPChemNet:
    enc_cfg = dict(cfg["model"]["encoder"])
    embed_dim = int(enc_cfg.get("embed_dim", 768))
    encoder = build_encoder(enc_cfg)
    embed_dim = int(getattr(encoder, "out_dim", embed_dim))
    trunk = cfg["model"]["trunk"]
    heads = cfg["model"].get("heads") or {}
    in_norm = cfg["model"].get("input_norm")
    if in_norm not in (None, "none", "robust"):
        raise ValueError(f"model.input_norm must be none|robust, got {in_norm!r}")
    in_norm_dim = int(enc_cfg.get("in_dim", 0)) if in_norm == "robust" else 0
    if in_norm == "robust" and in_norm_dim <= 0:
        raise ValueError("model.input_norm: robust needs model.encoder.in_dim")
    return XPChemNet(encoder, trunk_dim=int(trunk["dim"]),
                     dropout=float(trunk["dropout"]),
                     n_atm=n_atm, n_chem=n_chem, embed_dim=embed_dim,
                     side=cfg["model"].get("side"), n_aux=n_aux,
                     chem_condition=str(heads.get("chem_condition", "none")),
                     cond_logvar=bool(heads.get("cond_logvar", False)),
                     aux_condition=bool(heads.get("aux_condition", True)),
                     input_norm_dim=in_norm_dim)
