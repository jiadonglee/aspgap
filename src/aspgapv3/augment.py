"""Noise-matched augmentation: degrade bright APOGEE stars to faint-star XP noise.

Purpose (Gaia DR4 prep): APOGEE stars are mostly G ~ 10-14 with tiny XP noise,
but the stars one wants labels for span G up to 17.65. The model must see
realistic faint-star noise on bright-star spectra whose labels are well known.

For a star with raw DR3 coefficients c (110 = BP|RP), magnitude G and its own
coefficient noise (sigma_own, in xp343_sigma units) we draw a target magnitude
G' >= G, then

    s     = 10^(-0.4 (G' - G))                       flux dimming factor
    c'    = s c + L_d z,   z ~ N(0, I)               L_d: donor Cholesky (abs units)
    sigma'^2 = s^2 sigma_own^2 + sigma_d^2           per pixel, absolute flux units
    side' : G -> G',  log10 f550 -> log10 (D c')_107,  log10 SNR550 -> from sigma'
    labels unchanged.

G' is drawn from the target distribution (empirical G histogram of the full XP
catalog, or uniform) restricted to [G, g_max]: choose a 0.1 mag bin among those
at or above G with probability proportional to the target pmf, then uniform
within the bin, then max(G, .). The donor is a random bank star from that bin.

APPROXIMATIONS (be aware):
 * The donor's coefficient covariance comes from a star with a different SED
   (colour, line strength) than the target star: the noise level at G' is
   right on average, the pixel pattern of the noise is only approximately.
 * The donor noise is not rescaled within its 0.1 mag bin (<= 4.6% flux).
 * The donor noise is added at coefficient level (Gaussian, full BP & RP
   covariances); the star's own observed noise realization is kept and scaled by s.
 * xp343_sigma treats the 550nm normalizer as noise-free (sigma_abs / f_550);
   sigma' is divided by the *realized* f550' of the augmented star.
 * bp_rp (side) is left unchanged; stars whose 550nm flux turns non-positive
   after the noise draw, or without a valid covariance, are left undegraded.
 * Stars with G >= g_max are never degraded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data.xp import N_COEF, NORM_INDEX, coef_to_aspgap_input
from .data.xpnoise import pixel_sigma_abs, unpack_star

G_LO = 10.0      # first bank bin edge
DG = 0.1


class NoiseMatchAugment:
    def __init__(self, dm: dict, bank: dict, device: str = "cuda", p: float = 0.5,
                 g_target: str = "catalog_hist", g_max: float = 17.65):
        self.device = torch.device(device)
        self.p, self.g_max, self.g_target = float(p), float(g_max), g_target
        f64 = dict(dtype=torch.float64, device=self.device)
        self.dm_bp = torch.as_tensor(dm["dm_bp"], **f64)
        self.dm_rp = torch.as_tensor(dm["dm_rp"], **f64)
        self.w_bp = torch.as_tensor(dm["w_bp"], **f64)
        self.w_rp = torch.as_tensor(dm["w_rp"], **f64)

        chol = np.asarray(bank["chol"], np.float32)
        L_bp, L_rp = unpack_star(chol)
        self.L = torch.as_tensor(np.stack([L_bp, L_rp], 1), device=self.device)  # (nb,2,55,55)
        self.sig_d = torch.as_tensor(pixel_sigma_abs(chol, dm), **f64)            # (nb,343)
        edges = np.asarray(bank["bin_edges"], float)
        self.n_bins = len(edges) - 1
        self.bin_lo = torch.as_tensor(edges[:-1], **f64)
        self.bin_start = torch.as_tensor(np.asarray(bank["bin_start"]), device=self.device)
        self.bin_count = torch.as_tensor(np.asarray(bank["bin_count"]), device=self.device)
        # target pmf over bank bins
        if g_target == "catalog_hist":
            he = np.asarray(bank["hist_edges"], float)
            hc = np.asarray(bank["hist_counts"], float)
            idx = np.round((edges[:-1] - he[0]) / (he[1] - he[0])).astype(int)
            pmf = hc[idx]
        elif g_target == "uniform":
            pmf = np.ones(self.n_bins)
        else:
            raise ValueError(f"g_target must be catalog_hist|uniform, got {g_target!r}")
        pmf = pmf * (self.bin_count.cpu().numpy() > 0) * (edges[:-1] < self.g_max)
        self.pmf = torch.as_tensor(pmf / pmf.sum(), **f64)

    @classmethod
    def from_root(cls, root: str | Path, **kw) -> "NoiseMatchAugment":
        from .data.xp import build_design_matrix
        root = Path(root)
        dm = build_design_matrix(root / "xp_design_343.npz")
        with np.load(root / "noise_bank.npz") as z:
            bank = {k: z[k] for k in z.files}
        return cls(dm, bank, **kw)

    # -- sampling ----------------------------------------------------------
    def _rand(self, shape, gen):
        return torch.rand(shape, generator=gen, device=self.device, dtype=torch.float64)

    def draw_target(self, g: torch.Tensor, gen=None, fixed_g=None):
        """Returns (g_prime (B,), bin (B,), can (B,) bool)."""
        B = len(g)
        ar = torch.arange(self.n_bins, device=self.device)
        if fixed_g is not None:
            gp = torch.as_tensor(fixed_g, dtype=torch.float64, device=self.device).expand(B).clone()
            b = ((gp - G_LO) / DG).floor().long().clamp(0, self.n_bins - 1)
            can = (self.bin_count[b] > 0) & (g < gp) & (gp < self.g_max + 1e-9)
            return gp, b, can
        bG = ((g - G_LO) / DG).floor().long().clamp(0, self.n_bins - 1)
        w = self.pmf[None] * (ar[None] >= bG[:, None])
        can = (w.sum(1) > 0) & (g < self.g_max)
        w = torch.where(can[:, None], w, torch.ones_like(w))
        b = torch.multinomial(w, 1, generator=gen).squeeze(1)
        gp = self.bin_lo[b] + DG * self._rand((B,), gen)
        gp = torch.maximum(gp, g).clamp(max=self.g_max)
        return gp, b, can

    # -- main ----------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, batch: dict, gen: torch.Generator | None = None,
                 fixed_g: float | torch.Tensor | None = None,
                 p: float | None = None) -> dict:
        """batch needs coef (B,110), sigma (B,343), side (B,4: g, bp_rp,
        log_flux550, log_snr550), valid (B,). Returns a shallow copy with coef,
        sigma, side replaced for degraded rows and `aug` (B,) bool added.
        fixed_g: degrade every star fainter-than-nothing to exactly this G'
        (stars with G >= G' are left unchanged) -- used by evaluation."""
        dev = self.device
        coef = batch["coef"].to(dev).double()
        sig_own = batch["sigma"].to(dev).double()
        side = batch["side"].to(dev).double()
        g = side[:, 0]
        B = len(g)
        gp, b, can = self.draw_target(g, gen, fixed_g)
        pp = self.p if p is None else p
        sel = can & batch["valid"].to(dev) & torch.isfinite(coef).all(1) \
            & torch.isfinite(g) & torch.isfinite(side[:, 2])
        if fixed_g is None:
            sel &= self._rand((B,), gen) < pp
        if not bool(sel.any()):
            return {**batch, "aug": torch.zeros(B, dtype=torch.bool, device=dev)}

        s = 10.0 ** (-0.4 * (gp - g))                                   # (B,)
        j = self.bin_start[b] + (self._rand((B,), gen) * self.bin_count[b]).long()
        j = torch.minimum(j, self.bin_start[b] + self.bin_count[b] - 1)
        z = torch.randn((B, 2, N_COEF), generator=gen, device=dev, dtype=torch.float32)
        noise = torch.einsum("bkij,bkj->bki", self.L[j], z).double()    # (B,2,55)
        c = s[:, None] * coef
        c_bp = c[:, :N_COEF] + noise[:, 0]
        c_rp = c[:, N_COEF:] + noise[:, 1]
        flux = (c_bp @ self.dm_bp) * self.w_bp + (c_rp @ self.dm_rp) * self.w_rp
        f550 = flux[:, NORM_INDEX]
        sel &= torch.isfinite(f550) & (f550 > 0)

        f550_own = 10.0 ** side[:, 2]
        sig_abs = torch.sqrt((s[:, None] * sig_own * f550_own[:, None]) ** 2
                             + self.sig_d[j] ** 2)
        f550s = torch.where(sel, f550, torch.ones_like(f550))
        sig_new = sig_abs / f550s[:, None]
        side_new = side.clone()
        side_new[:, 0] = gp
        side_new[:, 2] = torch.log10(f550s)
        side_new[:, 3] = torch.log10(f550s / sig_abs[:, NORM_INDEX])
        m = sel[:, None]
        c_new = torch.cat([c_bp, c_rp], 1)
        out = dict(batch)
        out["coef"] = torch.where(m, c_new.float(), coef.float())
        # the network input x is NOT touched: the caller rebuilds it from out["coef"] / out["side"]
        # with apply_aspgap_input()
        out["sigma"] = torch.where(m, sig_new.float(), sig_own.float())
        out["side"] = torch.where(m, side_new.float(), side.float())
        out["aug"] = sel
        return out


def apply_aspgap_input(batch: dict, aspgap_sc: dict, device) -> torch.Tensor:
    """Network input `x` (B,110) for the coef110n baseline after augmentation.

    Degraded rows (batch["aug"]) get the AspGap forward transform of their
    degraded raw coefficients at the new G' (side col 0); undegraded rows keep
    the cached coef110n `x` verbatim. Shared by train.py (training batches and
    the degraded val set) and evaluate.py (degraded-G' mode)."""
    xa = coef_to_aspgap_input(batch["coef"].to(device), batch["side"][:, 0].to(device), aspgap_sc)
    return torch.where(batch["aug"].to(device)[:, None], xa, batch["x"].to(device))
