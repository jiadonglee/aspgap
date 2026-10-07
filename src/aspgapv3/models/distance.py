"""Absolute magnitude M_G / extinction A0 outputs + the physics that trains them.

DESIGN (aspgap_dist; everything here is off unless `model.distance.enabled`):

* DistanceHead: z (trunk feature) -> (M_G, log sigma^2_MG, A0).  M_G is the INTRINSIC absolute G
  mag.  A0 (extinction at 550 nm) = softplus(.) >= 0.
* NO apparent magnitude (or any brightness) reaches the network: the input stays the
  G-normalized coefficients, and the apparent G only enters the *loss* through the parallax
  likelihood.  M_G must therefore come from spectral shape, not from a Malmquist shortcut
  "faint star -> luminous".  (Residual caveat: with natural, un-augmented noise the XP noise level
  weakly encodes G; noise-matched augmentation decorrelates it.)
* Parallax likelihood (all stars with a parallax, no S/N or sign cut):
      plx_pred[mas] = 10^((10 - G + M_G + A_G)/5),   A_G = K["G"] * A0
      sigma^2       = sigma_plx^2 + (plx_pred * ln10/5 * sigma_MG)^2
      NLL           = 0.5 [ (plx_obs - zp - plx_pred)^2 / sigma^2 + ln sigma^2 ]
  G / plx / zero-point are those of the ORIGINAL star even when the input is noise-degraded
  (M_G, A0 are intrinsic).
* Photometric forward model: IntrinsicColourNet maps ONLY (Teff, logg, [M/H], [alpha/M]) (the
  normalized label vector -- never XP features) to the 7 intrinsic colours
  (BP-G, G-RP, G-J, G-H, G-K, G-W1, G-W2); observed colour = intrinsic + (K[a]-K[b]) * A0.
  Training feeds the APOGEE labels (teacher forcing; a missing label is filled with the network's
  own detached prediction); inference feeds the predicted labels.  Because the colour net
  cannot see the spectrum, a reddening of the XP/colour data can only be explained by A0.

EXTINCTION COEFFICIENTS K_X = A_X / A0 (A0 = A_550nm ~ A_V): fixed, Teff-independent, from
Wang & Chen (2019, ApJ 877, 116, Table 3, A_lambda/A_V with the Fitzpatrick/Schlafly-like
R_V = 3.1 law) : G 0.789, BP 1.002, RP 0.589, J 0.243, H 0.131, Ks 0.078, W1 0.039, W2 0.026.
The true Gaia coefficients depend on Teff and A0 (Riello+2021: k_G ~ 0.80-0.85 for
cool stars at low A0); this is ignored for now (k_G error of 0.04 -> 0.04 A0 mag in M_G).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2")      # column order of the magnitude arrays
EXT_K = {"G": 0.789, "BP": 1.002, "RP": 0.589, "J": 0.243, "H": 0.131,
         "K": 0.078, "W1": 0.039, "W2": 0.026}             # A_X / A0, Wang & Chen 2019
COLOURS = (("BP", "G"), ("G", "RP"), ("G", "J"), ("G", "H"), ("G", "K"), ("G", "W1"), ("G", "W2"))
COLOUR_NAMES = tuple(f"{a}-{b}" for a, b in COLOURS)
# d(observed colour)/dA0 = K[a] - K[b]
COLOUR_K = tuple(EXT_K[a] - EXT_K[b] for a, b in COLOURS)
LN10_5 = math.log(10.0) / 5.0
PLX_ZP_FALLBACK = -0.017   # mas, global DR3 zero-point, used where the Lindegren+21 recipe is undefined
PHOT_FLOOR = 0.02          # mag, added in quadrature to the catalogue photometric errors
MG_PRIOR = (2.0, 3.0)      # (mean, scale) used to de-normalize the M_G output (overridden at init from data)


class DistanceHead(nn.Module):
    """z (B, D) -> mu_mg (B,), logvar_mg (B,), a0 (B,) >= 0."""

    def __init__(self, in_dim: int, mg_mean: float = MG_PRIOR[0], mg_scale: float = MG_PRIOR[1],
                 a0_init: float = 0.1, sigma_mg_init: float = 0.3):
        super().__init__()
        self.out = nn.Linear(in_dim, 3)
        self.register_buffer("mg_mean", torch.tensor(float(mg_mean)))
        self.register_buffer("mg_scale", torch.tensor(float(mg_scale)))
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.copy_(torch.tensor([0.0, math.log(sigma_mg_init ** 2),
                                              math.log(math.expm1(a0_init))]))

    def set_mg_stats(self, mean: float, scale: float) -> None:
        self.mg_mean.fill_(float(mean))
        self.mg_scale.fill_(float(scale))

    def forward(self, z: torch.Tensor):
        with torch.autocast(device_type=z.device.type, enabled=False):    # fp32: bf16 would quantize M_G by ~0.02 mag
            o = self.out(z.float())
        mg = o[:, 0] * self.mg_scale + self.mg_mean
        return mg, o[:, 1], F.softplus(o[:, 2])


class IntrinsicColourNet(nn.Module):
    """(Teff, logg, [M/H], [alpha/M]) normalized -> intrinsic colours (B, 7).  Input is ONLY the
    n_in label values (default 4); there is deliberately no path from the spectrum.  Also holds a
    learned per-colour intrinsic scatter (extra variance on top of catalogue errors + floor)."""

    def __init__(self, n_in: int = 4, hidden: int = 128, n_col: int = len(COLOURS),
                 scatter_init: float = 0.03):
        super().__init__()
        self.n_in = n_in
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_col))
        self.log_sigma_int = nn.Parameter(torch.full((n_col,), math.log(scatter_init)))

    def set_output_bias(self, mean_colours) -> None:
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.as_tensor(mean_colours, dtype=torch.float32))

    def forward(self, labels_norm: torch.Tensor) -> torch.Tensor:
        if labels_norm.shape[-1] != self.n_in:
            raise ValueError(f"IntrinsicColourNet takes {self.n_in} label inputs, got {labels_norm.shape[-1]}")
        return self.net(labels_norm)


# ---------------------------------------------------------------- physics / losses

def parallax_pred(g: torch.Tensor, mg: torch.Tensor, a0: torch.Tensor,
                  k_g: float = EXT_K["G"]) -> torch.Tensor:
    """Predicted parallax [mas]: 10^((10 - G + M_G + A_G)/5), A_G = k_g * A0."""
    return 10.0 ** ((10.0 - g + mg + k_g * a0) / 5.0)


def parallax_nll(plx_obs, plx_err, zp, g, mg, logvar_mg, a0, ok, clamp=(-8.0, 8.0), k_g: float = EXT_K["G"]):
    """Mean Gaussian NLL over rows with ok (all tensors (B,), fp32).  plx_obs raw DR3 parallax,
    zp the Lindegren zero-point (corrected parallax = plx_obs - zp).  Returns (nll, pull (B,))."""
    pp = parallax_pred(g, mg, a0, k_g)
    s_mg = logvar_mg.clamp(*clamp).mul(0.5).exp()
    var = plx_err ** 2 + (pp * LN10_5 * s_mg) ** 2 + 1e-12
    d = (plx_obs - zp) - pp
    nll = 0.5 * (d ** 2 / var + torch.log(var))
    w = ok.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0), d / var.sqrt()


def colour_nll(col_obs, col_err, col_ok, col_int, a0, log_sigma_int, floor: float = PHOT_FLOOR):
    """Mean Gaussian NLL over valid (star, colour) entries.  col_* (B, 7); col_int the intrinsic
    colours; the reddening shift is COLOUR_K * A0."""
    k = torch.as_tensor(COLOUR_K, dtype=col_int.dtype, device=col_int.device)
    pred = col_int + k[None] * a0[:, None]
    var = col_err ** 2 + floor ** 2 + (2.0 * log_sigma_int).exp()[None] + 1e-12
    nll = 0.5 * ((col_obs - pred) ** 2 / var + torch.log(var))
    w = col_ok.to(nll.dtype)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def colour_inputs(y_norm: torch.Tensor, mask: torch.Tensor, mu_pred: torch.Tensor, n_in: int = 4):
    """Teacher forcing: true normalized labels where present, else the (detached) prediction."""
    y, m, mu = y_norm[:, :n_in], mask[:, :n_in], mu_pred[:, :n_in].detach()
    return torch.where(m, y, mu)


def distance_losses(net, out: dict, batch: dict, device, clamp=(-8.0, 8.0)) -> dict:
    """Parallax and photometric NLLs for one batch.  `out` are the (fp32) network outputs for the
    (possibly degraded) input; the astro columns of `batch` always belong to the original star."""
    a = batch["astro"]
    g = a["g"].to(device)
    l_plx, pull = parallax_nll(a["plx"].to(device), a["plx_err"].to(device), a["zp"].to(device), g,
                               out["mg"], out["logvar_mg"], out["a0"], a["plx_ok"].to(device), clamp)
    mu_all = torch.cat([out["mu_atm"], out["mu_chem"]], 1)
    cin = colour_inputs(batch["y"].to(device), batch["mask"].to(device), mu_all, net.colour_net.n_in)
    col_int = net.colour_net(cin)
    l_phot = colour_nll(a["col"].to(device), a["col_err"].to(device), a["col_ok"].to(device),
                        col_int, out["a0"], net.colour_net.log_sigma_int)
    return {"plx": l_plx, "phot": l_phot, "pull": pull}
