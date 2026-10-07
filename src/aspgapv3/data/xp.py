"""Gaia XP coefficients -> SpecCLIP-compatible 343-pixel input spectra.

Convention (cross-checked against SpecCLIP's own downstream code
`downstream-with-demo/predict_xp_wclip_v0.py` and against the working adapter
in gaiaworld):

    grid        np.arange(336, 1021, 2) nm  ->  343 samples
    normalize   flux / flux[107]            (index 107 == 550 nm)
    tensor      [batch, 343, 1]

We evaluate the BP/RP coefficient system *directly* on that grid via the
gaiaxpy design matrices (the same machinery old AspGap used, see
../aspgap/src/xp_utils.py:  flux = coef @ dm, then merge BP/RP with per-pixel
weights). SpecCLIP linearly interpolated its sampled spectra onto the same
grid; evaluating the analytic curves at the grid points is the same operation
without the intermediate resample.

gaiaxpy is needed exactly once, to build the (55, 343) design matrices; they
are cached to disk and every later build is a matmul.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

XP_GRID_NM: np.ndarray = np.arange(336.0, 1021.0, 2.0)
assert XP_GRID_NM.size == 343
NORM_INDEX: int = 107                       # XP_GRID_NM[107] == 550.0 nm
N_COEF: int = 55                            # per photometer, DR3
N_COEF_TOTAL: int = 110                     # BP (first 55) + RP (last 55)


def design_matrix_cache_path(data_root: str | Path) -> Path:
    return Path(data_root) / "xp_design_343.npz"


def build_design_matrix(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Evaluate the gaiaxpy BP/RP design matrices on the SpecCLIP grid.

    Needs gaiaxpy only the first time; afterwards reads the npz cache.
    Returns dict with dm_bp, dm_rp (55, 343), w_bp, w_rp (343,), grid (343,).
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    try:
        from gaiaxpy.calibrator.calibrator import (  # type: ignore
            __generate_xp_matrices_and_merge as generate_xp_matrices_and_merge,
        )
    except ImportError as exc:
        raise ImportError(
            "building the XP design matrix needs gaiaxpy (once ever; the "
            "matrices are cached afterwards): pip install gaiaxpy"
        ) from exc

    xp_dm, xp_merge = generate_xp_matrices_and_merge(
        "calibrator", XP_GRID_NM, "v375wi", "v142r"
    )
    dm_bp = np.asarray(xp_dm["bp"].get_design_matrix(), dtype=np.float64)
    dm_rp = np.asarray(xp_dm["rp"].get_design_matrix(), dtype=np.float64)
    w_bp = np.asarray(xp_merge["bp"], dtype=np.float64)
    w_rp = np.asarray(xp_merge["rp"], dtype=np.float64)

    # old AspGap convention: flux = coef @ dm  =>  dm is (n_coef, n_wav)
    if dm_bp.shape[0] != N_COEF or dm_bp.shape[-1] != XP_GRID_NM.size:
        if dm_bp.shape[-1] == N_COEF:       # transposed layout
            dm_bp, dm_rp = dm_bp.T, dm_rp.T
        else:
            raise RuntimeError(f"unexpected design matrix shape {dm_bp.shape}")

    out = {"dm_bp": dm_bp, "dm_rp": dm_rp, "w_bp": w_bp, "w_rp": w_rp,
           "grid": XP_GRID_NM}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    return out


def absolute_flux(coef110: np.ndarray, dm: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 110) coefficients -> (n, 343) BP/RP-merged flux in the design
    matrix's physical units (~1e-19 W nm-1 m-2), float64, NOT normalized."""
    c = np.atleast_2d(np.asarray(coef110, dtype=np.float64))
    if c.shape[1] != N_COEF_TOTAL:
        raise ValueError(f"expected (n, {N_COEF_TOTAL}) coefficients, got {c.shape}")
    return (c[:, :N_COEF] @ dm["dm_bp"]) * dm["w_bp"] \
        + (c[:, N_COEF:] @ dm["dm_rp"]) * dm["w_rp"]


def sample_spectra(coef110: np.ndarray, dm: dict[str, np.ndarray]) -> np.ndarray:
    """(n, 110) coefficients -> (n, 343) BP/RP-merged, 550nm-normalized flux.

    Layout matches the old AspGap chunks: xp[:, :55] = BP, xp[:, 55:] = RP.
    Rows whose 550nm flux is ~0 are returned as zeros (build_dataset drops them
    and records the count).
    """
    flux = absolute_flux(coef110, dm)

    norm = flux[:, NORM_INDEX][:, None]
    # degenerate rows: non-finite, exactly zero, or all-zero spectra. The
    # design matrix carries physical-flux normalization (~1e-19 W nm-1 m-2),
    # so never use an absolute threshold here; the 550nm ratio is unit-free.
    bad = ~np.isfinite(norm) | (norm == 0) | (np.max(np.abs(flux), 1) == 0)[:, None]
    norm = np.where(bad, 1.0, norm)
    out = (flux / norm).astype(np.float32)
    out[bad[:, 0]] = 0.0
    return out


def split_coef110(coef110: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return coef110[:, :N_COEF], coef110[:, N_COEF:]


def aspgap_scaler_arrays(root: str | Path,
                         bp_path: str = "../aspgap/model/scaler_bp_gmagand_0330.gz",
                         rp_path: str = "../aspgap/model/scaler_rp_gmagand_0330.gz"
                         ) -> dict[str, np.ndarray]:
    """RobustScaler center_/scale_ of the old AspGap input (BP, RP), cached as
    <root>/aspgap_scaler.npz so later runs need neither joblib nor sklearn."""
    cache = Path(root) / "aspgap_scaler.npz"
    if cache.is_file():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import joblib
    bp, rp = joblib.load(bp_path), joblib.load(rp_path)
    out = {"center": np.concatenate([bp.center_, rp.center_]),
           "scale": np.concatenate([bp.scale_, rp.scale_])}
    np.savez(cache, **out)
    return out


def coef_to_aspgap_input(coef110: np.ndarray, g: np.ndarray,
                         sc: dict[str, np.ndarray]) -> np.ndarray:
    """Forward transform of the old AspGap network input (exact inverse of
    build_dataset.invert_aspgap_coef): divide by 10^((15-G)*0.4), then
    (x - center) / scale per photometer. Works on numpy or torch tensors."""
    import torch
    if isinstance(coef110, torch.Tensor):                           # torch (numpy>=2 arrays also have .device)
        gn = coef110.double() / 10.0 ** ((15.0 - g.double()) * 0.4)[:, None]
        c = torch.as_tensor(sc["center"], dtype=torch.float64, device=gn.device)
        s = torch.as_tensor(sc["scale"], dtype=torch.float64, device=gn.device)
        return ((gn - c) / s).float()
    gn = np.asarray(coef110, np.float64) / 10.0 ** ((15.0 - np.asarray(g, np.float64)) * 0.4)[:, None]
    return ((gn - sc["center"]) / sc["scale"]).astype(np.float32)
