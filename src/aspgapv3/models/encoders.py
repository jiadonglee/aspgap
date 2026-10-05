"""Encoders. One interface: (B, in_dim) -> (B, embed_dim); embed_dim is 768
for every variant so the trunk/heads/loss comparison is like-for-like.

- SpecCLIPEncoder  wraps SpecFormerControl1_stats from the local SpecCLIP
  checkout (imported via the editable install in the jepa311 env). Frozen or
  partially fine-tuned (last-k transformer blocks + final layernorm).
- ScratchXPTransformer  random-init transformer on the 343-pixel input.
- CoeffMLPEncoder  the AspGap-style baseline on 110 BP/RP coefficients.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

SPECCLIP_EMBED_DIM = 768


def default_specclip_ckpt() -> Path | None:
    env = os.environ.get("ASPGAP_SPECCLIP_CKPT")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    # .../aspgap_v3/src/aspgapv3/models/encoders.py -> parents[4] = workspace root
    p = Path(__file__).resolve().parents[4] / "gaiaworld" / "specclip_assets" / "xp_encoder.ckpt"
    return p if p.is_file() else None


class AttnPool(nn.Module):
    """K learnable queries cross-attend over the 344 output tokens; the K
    attended 768-d vectors are layer-normed, concatenated and projected back to
    768 (the trunk input dim). Unlike mean pooling the queries can pick the
    stats token or individual wavelength regions (e.g. Ca triplet, Mg b)."""

    def __init__(self, dim: int = SPECCLIP_EMBED_DIM, n_queries: int = 4, n_heads: int = 2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, dim) * dim ** -0.5)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(n_queries * dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:        # (B,T,D)
        kv = self.norm_kv(tokens)
        q = self.q.unsqueeze(0).expand(len(tokens), -1, -1)
        o, _ = self.attn(q, kv, kv, need_weights=False)               # (B,K,D)
        return self.proj(self.norm_out(o).flatten(1))


class SpecCLIPEncoder(nn.Module):
    """Pretrained SpecCLIP Gaia-XP encoder, optionally fine-tunable.

    unfreeze_last_k = 0  -> frozen (Stage A); kept in eval() permanently
                    k>0  -> last k transformer blocks + final_layernorm train
                    -1   -> everything trains (Stage C; use tiny lr_enc)
    The reconstruction `head` of the backbone is never used.

    pooling:
      "mean"           SpecCLIP downstream convention: mean over all 344
                       tokens (stats token diluted 344:1) -> 768.
      "mean_plus_stats" concat(mean over all tokens, stats-token embedding)
                       -> 1536. The stats token carries the per-spectrum
                       mean/std of the 550nm-normalized flux, i.e. the broad
                       continuum shape that per-pixel standardization inside
                       the backbone otherwise removes; mean pooling dilutes
                       it. Use this for parameter regression.
      "attn"           AttnPool over the 344 tokens (`attn_queries`,
                       `attn_heads`) -> 768. Trained at the head LR.
    """

    def __init__(self, ckpt_path: str | os.PathLike | None = None,
                 unfreeze_last_k: int = 0, embed_dim: int = SPECCLIP_EMBED_DIM,
                 pooling: str = "mean", attn_queries: int = 4,
                 attn_heads: int = 2):
        super().__init__()
        try:
            from specclip.models import SpecFormerControl1_stats
        except ImportError as exc:
            raise ImportError(
                "specclip not importable; activate the jepa311 env or add the "
                "SpecCLIP checkout to PYTHONPATH"
            ) from exc
        if pooling not in ("mean", "mean_plus_stats", "attn"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.pool = AttnPool(SPECCLIP_EMBED_DIM, attn_queries, attn_heads) \
            if pooling == "attn" else None
        self._out_dim = 2 * SPECCLIP_EMBED_DIM if pooling == "mean_plus_stats" \
            else SPECCLIP_EMBED_DIM
        if embed_dim not in (SPECCLIP_EMBED_DIM, self._out_dim):
            raise ValueError("SpecCLIP embeddings are 768-d by construction")
        path = Path(ckpt_path) if ckpt_path else default_specclip_ckpt()
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "SpecCLIP xp_encoder.ckpt not found; set ASPGAP_SPECCLIP_CKPT"
            )
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = SpecFormerControl1_stats(**ckpt["hyper_parameters"])
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            raise RuntimeError(f"checkpoint missing weights: {sorted(missing)[:5]}")
        self.backbone = model
        self.unfreeze_last_k = int(unfreeze_last_k)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if unfreeze_last_k != 0:
            n = len(self.backbone.blocks)
            k = n if unfreeze_last_k < 0 else min(unfreeze_last_k, n)
            for blk in self.backbone.blocks[n - k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.backbone.final_layernorm.parameters():
                p.requires_grad_(True)
        self.frozen = unfreeze_last_k == 0
        if self.frozen:
            self.backbone.eval()

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:                 # frozen means frozen, incl. dropout
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != 343:
            raise ValueError(f"SpecCLIPEncoder expects (B, 343), got {tuple(x.shape)}")
        out = self.backbone(x.unsqueeze(-1))
        emb = out["embedding"]                   # (B, 344, 768)
        if self.pooling == "attn":
            return self.pool(emb)
        if self.pooling == "mean":
            return emb.mean(1)                   # SpecCLIP downstream pooling
        return torch.cat([emb.mean(1), emb[:, 0, :]], dim=-1)


class ScratchXPTransformer(nn.Module):
    """Random-init control: pointwise tokenization + small transformer."""

    def __init__(self, embed_dim: int = SPECCLIP_EMBED_DIM, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 unfreeze_last_k: int = -1, **_):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = nn.Embedding(343, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, embed_dim)
        if unfreeze_last_k != -1:                # scratch models train fully
            raise ValueError("scratch encoder trains fully; unfreeze_last_k=-1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.in_proj(x.unsqueeze(-1)) + self.pos.weight[None, :343]
        h = self.blocks(t).mean(1)
        return self.out(self.norm(h))


class CoeffMLPEncoder(nn.Module):
    """AspGap-style baseline: 110 BP/RP coefficients -> MLP -> 768."""

    def __init__(self, embed_dim: int = SPECCLIP_EMBED_DIM, hidden: int = 512,
                 dropout: float = 0.1, unfreeze_last_k: int = -1, **_):
        super().__init__()
        if unfreeze_last_k != -1:
            raise ValueError("coeff_mlp encoder trains fully; unfreeze_last_k=-1")
        self.net = nn.Sequential(
            nn.Linear(110, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    table = {"specclip": SpecCLIPEncoder, "scratch": ScratchXPTransformer,
             "coeff_mlp": CoeffMLPEncoder}
    if kind not in table:
        raise ValueError(f"encoder.type must be one of {sorted(table)}")
    return table[kind](**cfg)
