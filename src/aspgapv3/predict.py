"""Inference API: Gaia DR3 XP coefficients + G -> Teff, logg, [M/H], [alpha/M], M_G, A0 and distance.

    from aspgapv3.predict import predict
    df = predict(bp_coefficients, rp_coefficients, phot_g_mean_mag)

Needs only numpy, pandas and torch.  The packaged weights file (weights/aspgap_v3_ext_r2.pt) carries the model,
the input scaler, the label normalisation and the distance constants.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models.net import XPChemNet, build_model

DEFAULT_WEIGHTS_NAME = "aspgap_v3_ext_r2.pt"
OUTPUT_COLUMNS = ["teff", "teff_err", "logg", "logg_err", "m_h", "m_h_err", "alpha_m", "alpha_m_err",
                  "mg", "mg_err", "a0", "dist_mod", "dist_mod_err", "distance_pc",
                  "teff_err_model", "logg_err_model", "m_h_err_model", "alpha_m_err_model"]
FLAG_COLUMNS = ["flag_bad_input", "flag_g_outside", "flag_teff_edge", "flag_high_sigma_teff",
                "flag_high_sigma_logg", "flag_high_sigma_m_h", "flag_high_sigma_alpha_m", "flag_high_sigma_mg"]


def default_weights_path() -> Path:
    """$ASPGAPV3_WEIGHTS, else <repo>/weights/ (editable install / clone), else <package>/weights/."""
    env = os.environ.get("ASPGAPV3_WEIGHTS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for d in (here.parents[2] / "weights", here.parent / "weights"):
        if (d / DEFAULT_WEIGHTS_NAME).is_file():
            return d / DEFAULT_WEIGHTS_NAME
    raise FileNotFoundError(f"{DEFAULT_WEIGHTS_NAME} not found next to the repository or package; "
                            "pass path=... or set ASPGAPV3_WEIGHTS")


class AspGapModel:
    """Network + normalisation constants loaded from the weights file."""

    def __init__(self, bundle: dict, device: str | torch.device = "cpu"):
        self.meta = {k: v for k, v in bundle.items() if k != "state_dict"}
        self.device = torch.device(device)
        mc = bundle["model_config"]
        self.net = build_model({"model": mc["model"]}, n_atm=mc["n_atm"], n_chem=mc["n_chem"])
        self.net.load_state_dict(bundle["state_dict"])      # strict
        self.net.to(self.device).eval()
        inp = bundle["input"]
        self.g_ref, self.g_exp = float(inp["g_ref"]), float(inp["g_exponent"])
        self.center = np.asarray(inp["center"], np.float64)
        self.scale = np.asarray(inp["scale"], np.float64)
        self.lab_mean = np.asarray(bundle["label_norm"]["mean"], np.float64)
        self.lab_std = np.asarray(bundle["label_norm"]["std"], np.float64)
        self.floor = np.array([bundle["label_floor"][n] for n in bundle["labels"]], np.float64)
        self.clamp = tuple(bundle["logvar_clamp"])
        self.k_g = float(bundle["distance"]["k_g"])
        self.domain = bundle["domain"]
        self.thresholds = bundle["sigma_thresholds"]
        self.version = bundle["version"]

    def __repr__(self) -> str:
        return f"AspGapModel({self.version}, device={self.device})"


def load_model(path: str | os.PathLike | None = None, device: str | torch.device = "cpu") -> AspGapModel:
    """Load the packaged (default) or a custom weights file."""
    path = Path(path) if path is not None else default_weights_path()
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    return AspGapModel(bundle, device)


def _as_matrix(col, n_coef: int, name: str) -> np.ndarray:
    """(N, 55) float array from a 2-D array, or an object/list column of arrays / '[..]' strings."""
    if isinstance(col, (pd.Series, pd.Index)):
        col = col.to_numpy()
    if isinstance(col, np.ndarray) and col.dtype != object and col.ndim == 2:
        out = col.astype(np.float64)
    else:
        rows = []
        for r in list(col):
            if r is None or (isinstance(r, float) and math.isnan(r)):
                rows.append(np.full(n_coef, np.nan))
                continue
            if isinstance(r, (str, bytes)):
                r = r.decode() if isinstance(r, bytes) else r
                toks = r.replace(",", " ").translate(str.maketrans("", "", "[](){}")).split()
                r = np.array([float(t) for t in toks], np.float64)
            rows.append(np.asarray(r, np.float64).ravel())
        try:
            out = np.stack(rows) if rows else np.zeros((0, n_coef))
        except ValueError as exc:
            raise ValueError(f"{name}: rows have inconsistent lengths") from exc
    if out.ndim != 2 or out.shape[1] != n_coef:
        raise ValueError(f"{name} must have shape (N, {n_coef}), got {out.shape}")
    return out


def prepare_input(model: AspGapModel, bp, rp, g) -> np.ndarray:
    """Raw coefficients + G -> network input (float64 -> float32): G-normalise then RobustScale."""
    coef = np.concatenate([bp, rp], 1)
    gn = coef / 10.0 ** ((model.g_ref - g) * model.g_exp)[:, None]
    return ((gn - model.center) / model.scale).astype(np.float32)


@torch.no_grad()
def predict(bp_coefficients, rp_coefficients, phot_g_mean_mag, batch_size: int = 8192,
            device: str | torch.device | None = None, model: AspGapModel | None = None,
            weights: str | os.PathLike | None = None) -> pd.DataFrame:
    """Predict stellar parameters, M_G, A0 and distance.

    bp_coefficients, rp_coefficients : (N, 55) arrays, or object/list columns (Gaia archive format).
    phot_g_mean_mag : (N,) Gaia DR3 G.  Rows with non-finite input give NaN outputs and flag_bad_input.
    device : "cpu", "cuda" or None (= cuda if available).  model / weights : preloaded model or weights path.

    Returns a DataFrame (index 0..N-1, same order) with OUTPUT_COLUMNS + FLAG_COLUMNS; *_err are 1-sigma
    predicted uncertainties.  Label *_err = sqrt(sigma_model^2 + floor^2) with the per-label training sigma floors
    (Teff 50 K, logg 0.05, [M/H] 0.02, [alpha/M] 0.02); *_err_model is the raw network sigma (excess over the
    floor).  mg_err is the network sigma (no floor in the distance loss); dist_mod_err = mg_err, A0 uncertainty
    is not included.
    """
    if model is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(weights, device)
    dev = model.device
    n_coef = int(model.meta["input"]["n_coef"]) // 2
    bp = _as_matrix(bp_coefficients, n_coef, "bp_coefficients")
    rp = _as_matrix(rp_coefficients, n_coef, "rp_coefficients")
    g = np.asarray(phot_g_mean_mag, np.float64).ravel()
    if not (len(bp) == len(rp) == len(g)):
        raise ValueError(f"length mismatch: bp {len(bp)}, rp {len(rp)}, G {len(g)}")
    n = len(g)

    good = np.isfinite(bp).all(1) & np.isfinite(rp).all(1) & np.isfinite(g)
    x = np.zeros((n, 2 * n_coef), np.float32)
    x[good] = prepare_input(model, bp[good], rp[good], g[good])

    mu = np.empty((n, 4), np.float64)
    lv = np.empty((n, 4), np.float64)
    mg = np.empty(n)
    lv_mg = np.empty(n)
    a0 = np.empty(n)
    for i in range(0, n, batch_size):
        out = model.net(torch.from_numpy(x[i:i + batch_size]).to(dev))
        mu[i:i + batch_size] = torch.cat([out["mu_atm"], out["mu_chem"]], 1).double().cpu().numpy()
        lv[i:i + batch_size] = torch.cat([out["logvar_atm"], out["logvar_chem"]], 1).double().cpu().numpy()
        mg[i:i + batch_size] = out["mg"].double().cpu().numpy()
        lv_mg[i:i + batch_size] = out["logvar_mg"].double().cpu().numpy()
        a0[i:i + batch_size] = out["a0"].double().cpu().numpy()

    lo, hi = model.clamp
    phys = mu * model.lab_std + model.lab_mean
    err_model = np.exp(0.5 * np.clip(lv, lo, hi)) * model.lab_std     # excess over the training sigma floor
    err = np.hypot(err_model, model.floor)                           # total: sqrt(model^2 + floor^2)
    mg_err = np.exp(0.5 * np.clip(lv_mg, lo, hi))
    dist_mod = g - mg - model.k_g * a0

    df = pd.DataFrame({
        "teff": phys[:, 0], "teff_err": err[:, 0], "logg": phys[:, 1], "logg_err": err[:, 1],
        "m_h": phys[:, 2], "m_h_err": err[:, 2], "alpha_m": phys[:, 3], "alpha_m_err": err[:, 3],
        "mg": mg, "mg_err": mg_err, "a0": a0, "dist_mod": dist_mod, "dist_mod_err": mg_err,
        "distance_pc": 10.0 ** (dist_mod / 5.0 + 1.0),
        "teff_err_model": err_model[:, 0], "logg_err_model": err_model[:, 1],
        "m_h_err_model": err_model[:, 2], "alpha_m_err_model": err_model[:, 3]})
    df.loc[~good, :] = np.nan

    th, dom = model.thresholds, model.domain
    flags = pd.DataFrame({
        "flag_bad_input": ~good,
        "flag_g_outside": good & (g > dom["g_max"]),
        "flag_teff_edge": good & ((df["teff"] > dom["teff_edge"]) | (df["teff_err"] > th["teff"])),
        "flag_high_sigma_teff": good & (df["teff_err"] > th["teff"]),
        "flag_high_sigma_logg": good & (df["logg_err"] > th["logg"]),
        "flag_high_sigma_m_h": good & (df["m_h_err"] > th["m_h"]),
        "flag_high_sigma_alpha_m": good & (df["alpha_m_err"] > th["alpha_m"]),
        "flag_high_sigma_mg": good & (df["mg_err"] > th["mg"]),
    })
    assert list(flags.columns) == FLAG_COLUMNS
    return pd.concat([df, flags], axis=1)
