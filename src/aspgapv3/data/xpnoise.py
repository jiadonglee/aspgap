"""Gaia XP coefficient noise: covariance, packed Cholesky factors, pixel sigma.

DR3 convention (identical to gaiaxpy `_correlation_to_covariance_dr3int5`,
used by ../aspgap/src/xp_utils.py XPSampler.sample), per photometer:

    corr   (55, 55)  from the 1485 stored lower-triangle (k=-1) entries,
                     unit diagonal, mirrored
    cov_gx = diag(e / std) @ corr @ diag(e / std)      (what gaiaxpy stores)
             e   = <band>_coefficient_errors (55,)
             std = <band>_standard_deviation (scalar LSQ scatter)
    C      = std^2 * cov_gx = diag(e) @ corr @ diag(e)  (physical covariance)

gaiaxpy's `_sample_error` (the flux_error of calibrate()) multiplies by `std`
AFTER propagating cov_gx, i.e. it uses C; we do the same, so the standard
deviation enters through the cov_gx convention and cancels analytically.
(../aspgap XPSampler.sample uses cov_gx without the std^2 factor: its pixel
errors/draws are low by the factor 1/std.)

BP and RP are independent. Per-pixel noise on the merged 343-pixel spectrum
f = w_bp (c_bp D_bp) + w_rp (c_rp D_rp):

    Var(f_p) = w_bp,p^2 [D_bp^T C_bp D_bp]_pp + w_rp,p^2 [D_rp^T C_rp D_rp]_pp
             = w^2 * || L^T D[:, p] ||^2          (C = L L^T)

NOTE: the gaiaxpy-derived XPSampler.sample multiplies the RP block by
w_rp[:,None]*w_rp[:,None] (a typo for the outer product); we use the correct
outer-product form, which has the same diagonal as gaiaxpy's own calibrate().

Storage: a lower-triangular 55x55 factor is packed row-major into 1540 floats
(np.tril_indices order); a star is [BP 1540 | RP 1540] = 3080 float32.
"""

from __future__ import annotations

import numpy as np

from .xp import N_COEF, NORM_INDEX

N_TRI = N_COEF * (N_COEF + 1) // 2          # 1540
N_PACK = 2 * N_TRI                          # 3080 per star (BP + RP)
_TRIL = np.tril_indices(N_COEF)             # (rows, cols), row-major
_TRIL_STRICT = np.tril_indices(N_COEF, k=-1)
JITTER_LEVELS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)   # x mean(diag cov)


def corr_to_cov_dr3(corr_flat: np.ndarray, err: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Batched. corr_flat (n,1485), err (n,55), std (n,) -> physical coefficient
    covariance C = std^2 * cov_gx (n,55,55). Rows with non-finite std (the
    gaiaxpy NaN case) come back all-NaN."""
    n = len(err)
    corr = np.zeros((n, N_COEF, N_COEF))
    corr[:, np.arange(N_COEF), np.arange(N_COEF)] = 1.0
    corr[:, _TRIL_STRICT[0], _TRIL_STRICT[1]] = corr_flat
    corr[:, _TRIL_STRICT[1], _TRIL_STRICT[0]] = corr_flat
    d = np.asarray(err, np.float64) / np.asarray(std, np.float64)[:, None]
    cov_gx = d[:, :, None] * corr * d[:, None, :]
    return cov_gx * (np.asarray(std, np.float64) ** 2)[:, None, None]


def cholesky_jitter(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched Cholesky with escalating diagonal jitter.

    Returns (L (n,55,55) float64, n_jitter_level (n,) int8 [0 = PD as is,
    k>0 = JITTER_LEVELS[k-1] x mean(diag) was added, -1 = failed or non-finite
    input -> L is zeros], ok (n,) bool)."""
    n = len(cov)
    L = np.zeros_like(cov)
    lvl = np.full(n, -1, np.int8)
    finite = np.isfinite(cov).all((1, 2))
    try:
        L[finite] = np.linalg.cholesky(cov[finite])
        lvl[finite] = 0
    except np.linalg.LinAlgError:
        for i in np.where(finite)[0]:
            C = cov[i]
            tries = [0.0] + [j * np.mean(np.diag(C)) for j in JITTER_LEVELS]
            for k, jit in enumerate(tries):
                try:
                    L[i] = np.linalg.cholesky(C + jit * np.eye(N_COEF))
                    lvl[i] = k
                    break
                except np.linalg.LinAlgError:
                    continue
    return L, lvl, lvl >= 0


def pack_tril(L: np.ndarray) -> np.ndarray:
    """(..., 55, 55) lower-triangular -> (..., 1540)."""
    return L[..., _TRIL[0], _TRIL[1]]


def unpack_tril(p: np.ndarray) -> np.ndarray:
    """(..., 1540) -> (..., 55, 55) lower-triangular (zeros above diagonal)."""
    out = np.zeros(p.shape[:-1] + (N_COEF, N_COEF), dtype=p.dtype)
    out[..., _TRIL[0], _TRIL[1]] = p
    return out


def pack_star(L_bp: np.ndarray, L_rp: np.ndarray) -> np.ndarray:
    return np.concatenate([pack_tril(L_bp), pack_tril(L_rp)], -1)


def unpack_star(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return unpack_tril(packed[..., :N_TRI]), unpack_tril(packed[..., N_TRI:])


def pixel_sigma_abs(packed: np.ndarray, dm: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 3080) packed factors -> (n, 343) 1-sigma of the merged flux in the
    design matrix's absolute units (matches gaiaxpy calibrate() flux_error)."""
    L_bp, L_rp = unpack_star(np.asarray(packed, np.float64))
    var = np.zeros((len(packed), dm["dm_bp"].shape[1]))
    for L, dmx, w in ((L_bp, dm["dm_bp"], dm["w_bp"]), (L_rp, dm["dm_rp"], dm["w_rp"])):
        A = np.swapaxes(L, 1, 2) @ dmx                      # (n, 55, 343) = L^T D
        var += (A ** 2).sum(1) * w[None] ** 2
    return np.sqrt(var)


def normalized_sigma_ratio(packed: np.ndarray, coef110: np.ndarray,
                           dm: dict[str, np.ndarray]) -> np.ndarray:
    """First-order sigma of the *ratio* f_p / f_550 (the true noise of xp343),
    with the 550nm normalizer's own noise and its correlation with f_p included:
    Var(r_p) = Var(f_p - r_p f_550) / f_550^2. Used only in tests / diagnostics;
    the stored xp343_sigma is the simpler sigma_abs / f_550 (see build script)."""
    from .xp import absolute_flux
    f = absolute_flux(coef110, dm)
    r = f / f[:, NORM_INDEX:NORM_INDEX + 1]
    L_bp, L_rp = unpack_star(np.asarray(packed, np.float64))
    var = np.zeros_like(r)
    for L, dmx, w in ((L_bp, dm["dm_bp"], dm["w_bp"]), (L_rp, dm["dm_rp"], dm["w_rp"])):
        # linear functional per pixel p: v_p = w_p D[:,p] - r_p w_107 D[:,107]
        V = w[None, None, :] * dmx[None] - r[:, None, :] * (w[NORM_INDEX] * dmx[:, NORM_INDEX])[None, :, None]
        A = np.swapaxes(L, 1, 2) @ V                        # (n,55,343)
        var += (A ** 2).sum(1)
    return np.sqrt(var) / np.abs(f[:, NORM_INDEX:NORM_INDEX + 1])


# ---------------------------------------------------------------- raw DR3 h5

def parse_h5_range(name: str) -> tuple[int, int]:
    """XpContinuousMeanSpectrum_000000-003111.h5 -> (0, 3111): inclusive
    range of source_id >> 43 (HEALPix level-8 index) held by the file."""
    lo, hi = name.rsplit("_", 1)[1].split(".")[0].split("-")
    return int(lo), int(hi)


def cov_from_h5_rows(h, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Read `rows` (sorted ints) from an open raw XP h5 and return float64
    coefficients, plus packed Cholesky (n,3080 float32), jitter levels (n,2),
    ok (n,) -- ok False if either band is not PD even with jitter / NaN std."""
    out: dict[str, np.ndarray] = {}
    Ls, lv = [], []
    coef = []
    for b in ("bp", "rp"):
        # per-row reads are far faster than h5py fancy indexing here
        g = lambda k: np.stack([h[f"{b}_{k}"][int(r)] for r in rows]) if len(rows) \
            else np.zeros((0,))
        c = g("coefficients").astype(np.float64)
        e = g("coefficient_errors").astype(np.float64)
        cr = g("coefficient_correlations").astype(np.float64)
        sd = np.array([h[f"{b}_standard_deviation"][int(r)] for r in rows], np.float64)
        cov = corr_to_cov_dr3(cr, e, sd)
        L, level, ok = cholesky_jitter(cov)
        coef.append(c)
        Ls.append((L, ok))
        lv.append(level)
    out["coef"] = np.concatenate(coef, 1)                    # (n,110) BP|RP
    out["chol"] = pack_star(Ls[0][0], Ls[1][0]).astype(np.float32)
    out["level"] = np.stack(lv, 1)
    out["ok"] = Ls[0][1] & Ls[1][1] & np.isfinite(out["coef"]).all(1)
    return out
