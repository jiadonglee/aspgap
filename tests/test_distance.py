"""M_G / A0 head, parallax likelihood, photometric forward model (models/distance.py)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aspgapv3.config import load_config
from aspgapv3.data.dataset import APOGEEDataset
from aspgapv3.dist_eval import distance_tables, parallax_arrays, red_clump
from aspgapv3.models.distance import (BANDS, COLOUR_K, COLOURS, EXT_K, IntrinsicColourNet, colour_inputs,
                                      colour_nll, distance_losses, parallax_nll, parallax_pred)
from aspgapv3.models.net import build_model
from test_p1 import _toy_root, scratch_cfg

CFG = Path(__file__).resolve().parents[1] / "configs"


def _cfg(dist=True):
    c = scratch_cfg()
    if dist:
        c["model"]["distance"] = {"enabled": True, "colour_hidden": 16}
    return c


def test_parallax_formula():
    # M_G = 5, G = 15, no extinction: d = 10^((15-5+5)/5) = 1000 pc -> 1 mas
    assert float(parallax_pred(torch.tensor(15.0), torch.tensor(5.0), torch.tensor(0.0))) == pytest.approx(1.0)
    # 1 mag of G extinction at fixed M_G, G: parallax grows by 10^(0.2) (the star must be closer)
    a0 = 1.0 / EXT_K["G"]
    assert float(parallax_pred(torch.tensor(15.0), torch.tensor(5.0), torch.tensor(a0))) == pytest.approx(10 ** 0.2)
    # distance modulus consistency: mu = G - M_G - A_G, plx = 1000 / 10^((mu+5)/5)
    g, mg, a = 13.3, 1.1, 0.4
    mu = g - mg - EXT_K["G"] * a
    assert float(parallax_pred(torch.tensor(g), torch.tensor(mg), torch.tensor(a))) == pytest.approx(
        1000.0 / 10 ** ((mu + 5) / 5))


def test_parallax_nll_value_masking_and_zero_point():
    g, mg, a0 = torch.tensor([15.0, 15.0]), torch.tensor([5.0, 5.0]), torch.zeros(2)
    lv = torch.full((2,), math.log(0.1 ** 2))
    err = torch.tensor([0.1, 0.1])
    # star 0: obs - zp = 1.3 vs pred 1.0
    zp = torch.tensor([-0.3, 0.0])
    obs = torch.tensor([1.0, 123.0])
    nll, pull = parallax_nll(obs, err, zp, g, mg, lv, a0, torch.tensor([True, False]))
    sig2 = 0.1 ** 2 + (1.0 * math.log(10) / 5 * 0.1) ** 2
    assert float(nll) == pytest.approx(0.5 * (0.3 ** 2 / sig2 + math.log(sig2)), rel=1e-5)   # star 1 masked out
    assert float(pull[0]) == pytest.approx(0.3 / math.sqrt(sig2), rel=1e-5)
    # negative parallaxes are legal observations (no sign cut)
    n2, _ = parallax_nll(torch.tensor([-0.2]), torch.tensor([0.3]), torch.tensor([-0.02]), torch.tensor([17.0]),
                         torch.tensor([3.0]), torch.zeros(1), torch.zeros(1), torch.tensor([True]))
    assert torch.isfinite(n2)
    # gradient reaches M_G, sigma_MG and A0
    mg_ = torch.tensor([4.0], requires_grad=True)
    lv_ = torch.zeros(1, requires_grad=True)
    a_ = torch.tensor([0.2], requires_grad=True)
    parallax_nll(torch.tensor([1.0]), torch.tensor([0.1]), torch.zeros(1), torch.tensor([15.0]), mg_, lv_, a_,
                 torch.tensor([True]))[0].backward()
    assert all(t.grad is not None and t.grad.abs().item() > 0 for t in (mg_, lv_, a_))


def test_colour_forward_model_shapes_masking_and_reddening():
    assert [f"{a}-{b}" for a, b in COLOURS] == ["BP-G", "G-RP", "G-J", "G-H", "G-K", "G-W1", "G-W2"]
    # BP-G gets redder (more positive) with A0: k_BP > k_G ; G-K gets bluer (negative): k_G - k_K > 0 -> redder?
    k = dict(zip(COLOURS, COLOUR_K))
    assert k[("BP", "G")] == pytest.approx(EXT_K["BP"] - EXT_K["G"]) and k[("BP", "G")] > 0
    assert k[("G", "K")] == pytest.approx(EXT_K["G"] - EXT_K["K"]) and k[("G", "K")] > 0
    B = 3
    col_int = torch.randn(B, 7)
    a0 = torch.tensor([0.0, 0.5, 2.0])
    obs = col_int + torch.tensor(COLOUR_K)[None] * a0[:, None]             # perfect model
    err, ok = torch.full((B, 7), 0.01), torch.ones(B, 7, dtype=torch.bool)
    ls = torch.full((7,), -20.0)
    perfect = colour_nll(obs, err, ok, col_int, a0, ls)
    wrong = colour_nll(obs, err, ok, col_int, a0 + 1.0, ls)
    assert wrong > perfect
    # masked entries do not contribute even when wildly wrong (nan-safe: values are finite placeholders)
    bad = obs.clone()
    bad[1, 4:] += 100.0
    okm = ok.clone()
    okm[1, 4:] = False
    assert colour_nll(bad, err, okm, col_int, a0, ls) == pytest.approx(float(perfect), abs=1e-6)
    assert colour_nll(obs, err, torch.zeros_like(ok), col_int, a0, ls) == 0.0     # nothing observed
    # floor: tiny catalogue errors cannot give a smaller variance than 0.02^2
    n0 = colour_nll(obs + 0.02, torch.zeros(B, 7), ok, col_int, a0, ls)
    assert float(n0) == pytest.approx(0.5 * (1.0 + math.log(0.02 ** 2)), rel=1e-3)


def test_intrinsic_colour_net_never_sees_spectrum():
    net = IntrinsicColourNet(4, 16)
    assert net.net[0].in_features == 4 and net(torch.randn(5, 4)).shape == (5, 7)
    with pytest.raises(ValueError):
        net(torch.randn(5, 343))
    # in the full model the colour loss reaches the spectrum ONLY through A0, never through the colour net input
    m = build_model(_cfg(), n_atm=3, n_chem=1).eval()
    x = torch.rand(6, 343).requires_grad_(True)
    out = m(x)
    batch = {"y": torch.randn(6, 4), "mask": torch.ones(6, 4, dtype=torch.bool),
             "astro": {"g": torch.full((6,), 14.), "plx": torch.ones(6), "plx_err": torch.full((6,), .05),
                       "zp": torch.zeros(6), "plx_ok": torch.ones(6, dtype=torch.bool),
                       "col": torch.randn(6, 7), "col_err": torch.full((6, 7), .03),
                       "col_ok": torch.ones(6, 7, dtype=torch.bool)}}
    # (a) with A0 detached, the colour NLL has zero gradient w.r.t. the input spectrum
    o2 = dict(out)
    o2["a0"] = out["a0"].detach()
    distance_losses(m, o2, batch, "cpu")["phot"].backward()
    assert x.grad is None or float(x.grad.abs().sum()) == 0.0
    # (b) with A0 attached it does flow (through the A0 head)
    distance_losses(m, out, batch, "cpu")["phot"].backward()
    assert float(x.grad.abs().sum()) > 0


def test_teacher_forcing_fills_missing_labels_with_detached_prediction():
    y = torch.tensor([[1., 2., 3., 4.]])
    mu = torch.tensor([[9., 9., 9., 9.]], requires_grad=True)
    m = torch.tensor([[True, False, True, False]])
    cin = colour_inputs(y, m, mu)
    assert cin.tolist() == [[1., 9., 3., 9.]] and not cin.requires_grad


def test_distance_head_outputs_nonneg_a0_and_fp32():
    m = build_model(_cfg(), n_atm=3, n_chem=1)
    out = m(torch.rand(8, 343) + .5)
    assert out["mg"].shape == out["logvar_mg"].shape == out["a0"].shape == (8,)
    assert (out["a0"] >= 0).all() and out["mg"].dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(torch.rand(8, 343) + .5)["mg"].dtype == torch.float32


def test_default_off_and_state_dict_identical_without_distance():
    for p in sorted(CFG.glob("*.yaml")):
        has = bool((load_config(p, []).get("model") or {}).get("distance", {}).get("enabled", False))
        assert has == (p.stem == "aspgap_dist"), p.stem
    a = build_model(_cfg(False), n_atm=3, n_chem=3)
    assert not any("dist" in k or "colour" in k for k in a.state_dict())
    assert "mg" not in a(torch.rand(2, 343) + .5)
    cfg = load_config(CFG / "aspgap_dist.yaml", [])
    assert cfg["labels"]["atm"] + cfg["labels"]["chem"] == ["teff", "logg", "m_h", "alpha_m"]
    assert cfg["train"]["sampler"]["type"] == "random"
    assert load_config(CFG / "base.yaml", [])["train"]["sampler"]["type"] == "random"
    assert not cfg["model"]["encoder"].get("side") and not cfg["model"].get("side")   # no brightness input


def _toy_astro(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    cat = pd.read_parquet(root / "catalog.parquet").sort_values("row")
    t = pd.DataFrame({"row": cat["row"].to_numpy(), "source_id": cat["source_id"].to_numpy(),
                      "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n), "l": rng.uniform(0, 360, n),
                      "b": rng.uniform(-80, 80, n), "ecl_lat": 0.0,
                      "parallax": rng.normal(1, .3, n), "parallax_error": 0.05,
                      "phot_g_mean_mag": rng.uniform(10, 16, n), "phot_bp_mean_mag": 12.0,
                      "phot_rp_mean_mag": 11.0, "e_g": .003, "e_bp": .005, "e_rp": .005,
                      "j_m": 10.0, "j_msigcom": .02, "h_m": 9.5, "h_msigcom": .02, "ks_m": 9.4, "ks_msigcom": .02,
                      "w1mpro": 9.3, "w1mpro_error": .02, "w2mpro": 9.3, "w2mpro_error": .02,
                      "tmass_ok_j": True, "tmass_ok_h": True, "tmass_ok_k": True,
                      "wise_ok_w1": True, "wise_ok_w2": True, "zpt": -0.03})
    t.loc[3, "parallax"] = np.nan                       # no parallax
    t.loc[4, ["tmass_ok_j", "tmass_ok_k"]] = False      # quality-masked 2MASS
    t.loc[5, "w1mpro"] = np.nan
    t.to_parquet(root / "astrometry_phot.parquet")
    return t


def test_dataset_astro_alignment_masks_and_original_g(tmp_path):
    _toy_root(tmp_path, with_aux=False)
    t = _toy_astro(tmp_path)
    ds = APOGEEDataset(tmp_path, "train", labels=["teff", "logg", "m_h", "alpha_m"], astro=True)
    assert ds.labels == ["teff", "logg", "m_h", "alpha_m"] and ds[0]["y"].shape == (4,)
    assert ds.scaler.labels == ds.labels and len(ds.scaler.mean) == 4         # restricted scaler
    assert "astro" not in APOGEEDataset(tmp_path, "train")[0]
    for i in (0, 3, 4, 5):
        a, row = ds[i]["astro"], ds.indices[i]
        assert float(a["g"]) == pytest.approx(t["phot_g_mean_mag"][row], abs=1e-5)
        assert bool(a["plx_ok"]) == bool(np.isfinite(t["parallax"][row]))
        assert float(a["zp"]) == pytest.approx(-0.03)
    # J and K quality-masked -> G-J, G-K masked, the other colours kept; missing W1 -> only G-W1 masked
    i4 = int(np.where(ds.indices == 4)[0][0]) if 4 in ds.indices else None
    if i4 is not None:
        assert ds[i4]["astro"]["col_ok"].tolist() == [True, True, False, True, False, True, True]
    i5 = int(np.where(ds.indices == 5)[0][0]) if 5 in ds.indices else None
    if i5 is not None:
        assert ds[i5]["astro"]["col_ok"].tolist() == [True, True, True, True, True, False, True]
    with pytest.raises(FileNotFoundError, match="build_astrometry_phot"):
        (tmp_path / "astrometry_phot.parquet").unlink()
        APOGEEDataset(tmp_path, "train", astro=True)


def test_dist_eval_pull_and_resid_self_consistent():
    rng = np.random.default_rng(0)
    n = 4000
    g = rng.uniform(10, 16, n)
    mg = rng.uniform(-1, 5, n)
    a0 = rng.uniform(0, 0.5, n)
    pp = 10 ** ((10 - g + mg + EXT_K["G"] * a0) / 5)
    err = np.full(n, 0.03)
    smg = 0.2
    sig = np.sqrt(err ** 2 + (pp * math.log(10) / 5 * smg) ** 2)
    plx = pp + rng.normal(0, 1, n) * sig
    ar = {"g": g, "plx": plx - 0.03, "plx_err": err, "zp": np.full(n, -0.03), "plx_ok": np.ones(n, bool),
          "mag": np.zeros((n, 8)), "mag_ok": np.ones((n, 8), bool)}
    lv = np.full(n, math.log(smg ** 2))
    t, d = distance_tables(mg, lv, a0, ar)
    assert abs(t.loc[0, "pull_median"]) < 0.08 and abs(t.loc[0, "pull_rstd"] - 1) < 0.08
    assert t.loc[0, "MG_resid_rmse"] < 0.5
    # red clump helper returns M_K ~ consistent with the injected truth
    ar["mag"][:, BANDS.index("K")] = -1.6 + (g - mg - EXT_K["G"] * a0) + EXT_K["K"] * a0      # M_K = -1.6
    y = np.tile([4800.0, 2.45, 0.0, 0.0], (n, 1))
    rc = red_clump(y, np.ones((n, 4), bool), ["teff", "logg", "m_h", "alpha_m"], ar, mg, a0, np.ones(n, bool))
    assert rc["n"] == n and rc["MK_median"] == pytest.approx(-1.6, abs=1e-6) and rc["MK_std"] < 1e-5


def test_anchor_masks_targets_and_loss():
    from aspgapv3.models.distance import anchor_nll, anchor_targets
    #            near      near-lowSNR  highlat-far  highlat-near-plane  low-b far  neg plx   no SFD
    plx = np.array([20.0,   20.0,        1.0,         1.0,                1.0,       -1.0,     1.0])
    err = np.array([0.5,    5.0,         0.1,         0.1,                0.1,       0.1,      0.1])
    b = np.array([10.0,     10.0,        60.0,        10.0,               10.0,      60.0,     60.0])
    ebv = np.array([0.1,    0.1,         0.05,        0.05,               0.2,       0.05,     np.nan])
    # d*sin|b| : 1000*sin60 = 866 pc > 300 (row 2); 1000*sin10 = 174 pc < 300 (row 3)
    t, s, k = anchor_targets(plx, err, b, ebv)
    assert k.tolist() == [1, 0, 2, 0, 0, 0, 0]
    assert t[0] == 0 and s[0] == pytest.approx(0.03)
    assert t[2] == pytest.approx(3.1 * 0.86 * 0.05) and s[2] == pytest.approx(0.03 + 0.1 * t[2])
    a0 = torch.tensor([0.3, 5.0, 0.0])
    tg, sg = torch.tensor([0.0, 9.0, 0.2]), torch.tensor([0.03, 1.0, 0.04])
    ok = torch.tensor([True, False, True])
    expect = 0.5 * ((0.3 / 0.03) ** 2 + (0.2 / 0.04) ** 2) / 2
    assert float(anchor_nll(a0, tg, sg, ok)) == pytest.approx(expect)
    assert float(anchor_nll(a0, tg, sg, torch.zeros(3, dtype=torch.bool))) == 0.0


def test_dataset_anchors_train_only(tmp_path):
    _toy_root(tmp_path, with_aux=False)
    t = _toy_astro(tmp_path)
    t["parallax"] = 20.0
    t["parallax_error"] = 0.1
    t.to_parquet(tmp_path / "astrometry_phot.parquet")
    tr = APOGEEDataset(tmp_path, "train", astro=True)
    va = APOGEEDataset(tmp_path, "val", astro=True)
    assert (tr.anchor_set == 1).sum() > 30 and (va.anchor_set == 0).all()
    assert bool(tr[0]["astro"]["anc_ok"]) and not bool(va[0]["astro"]["anc_ok"])
