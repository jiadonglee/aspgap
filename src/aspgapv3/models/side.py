"""Side-input conditioning: XP noise level + absolute-flux scalars -> FiLM.

AspGap's two real advantages over a purely normalized-spectrum model are
(i) absolute flux / continuum information (G, flux at 550nm: lost when the
spectrum is divided by f_550) and (ii) the XP noise covariance. Both enter
here as a *side* vector that modulates the trunk features, never SpecCLIP's
internals (its per-spectrum normalization would erase them anyway):

    noise channel  r_p = sigma_p / max(|x_p|, 0.02)     (xp343_sigma / xp343)
                   -> log10 clamp(r, 1e-4, 1e2) -> (v + 1.5) / 1.0
                   -> 1D CNN (3 strided convs + adaptive pool + linear) -> 64-d
    scalars        standardized with train-split mean/std (buffers, saved in
                   the checkpoint); NaNs -> 0 after standardization
    FiLM           MLP([noise_vec, scalars]) -> (gamma, beta) in R^{2 x trunk};
                   z' = (1 + gamma) z + beta. The output layer is zero-initialized,
                   so at step 0 the model is exactly the side-less model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SIDE_SCALARS = ("g", "bp_rp", "log_flux550", "log_snr550")      # names in configs
SIDE_COLUMN = {"g": 0, "bp_rp": 1, "log_flux550": 2, "log_snr550": 3}  # side.npy cols


def relative_noise_feature(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """(B,343),(B,343) -> (B,343) log-relative-error feature, roughly O(1)."""
    rel = sigma.float() / x.float().abs().clamp_min(0.02)
    return (torch.log10(rel.clamp(1e-4, 1e2)) + 1.5)


class NoiseCNN(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Linear(32 * 8, out_dim)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.fc(self.conv(f.unsqueeze(1)).flatten(1)))


class SideFiLM(nn.Module):
    def __init__(self, trunk_dim: int, scalars: list[str], noise_channel: bool,
                 hidden: int = 128, noise_dim: int = 64):
        super().__init__()
        for s in scalars:
            if s not in SIDE_COLUMN:
                raise ValueError(f"unknown side scalar {s!r}; options {list(SIDE_COLUMN)}")
        self.scalars = list(scalars)
        self.cols = [SIDE_COLUMN[s] for s in scalars]
        self.noise_channel = bool(noise_channel)
        self.cnn = NoiseCNN(noise_dim) if noise_channel else None
        in_dim = len(scalars) + (noise_dim if noise_channel else 0)
        if in_dim == 0:
            raise ValueError("side conditioning needs scalars and/or noise_channel")
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 2 * trunk_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("s_mean", torch.zeros(len(scalars)))
        self.register_buffer("s_std", torch.ones(len(scalars)))

    def set_scalar_stats(self, mean, std) -> None:
        self.s_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.s_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, z: torch.Tensor, x: torch.Tensor, side: dict) -> torch.Tensor:
        feats = []
        if self.scalars:
            s = (side["scalars"].float() - self.s_mean) / self.s_std
            feats.append(torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).clamp(-8, 8))
        if self.noise_channel:
            feats.append(self.cnn(relative_noise_feature(x, side["sigma"])))
        gamma, beta = self.out(self.mlp(torch.cat(feats, -1))).chunk(2, -1)
        return (1.0 + gamma) * z + beta


def build_side(net, batch: dict) -> dict | None:
    """Pick the side inputs a model wants from a (possibly augmented) batch."""
    cond = getattr(net, "side", None)
    if cond is None or "side" not in batch:
        return None
    dev = next(net.parameters()).device
    return {"scalars": batch["side"].to(dev)[:, cond.cols] if cond.cols
            else batch["side"].to(dev)[:, :0],
            "sigma": batch["sigma"].to(dev)}
