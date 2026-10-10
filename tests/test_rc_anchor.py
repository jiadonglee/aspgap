"""Red-clump A0 anchors (rc_anchor.py): selection, self-calibration, anchor targets, dataset integration."""
import numpy as np
import pytest
import torch

from aspgapv3.models.distance import EXT_K, anchor_nll
from aspgapv3.rc_anchor import RC_SET, calibrate_rc, rc_anchor_targets, rc_mg, rc_select
from test_distance import _toy_astro
from test_p1 import _toy_root


def _fake_rc(n=4000, a0_true=0.0, seed=1, mg0=0.55, sig=0.12, plx_err=0.02):
    rng = np.random.default_rng(seed)
    m_h = rng.uniform(-0.6, 0.3, n)
    teff = 4607 - 382.5 * m_h + rng.uniform(-100, 150, n)
    logg = rng.uniform(2.3, 2.45, n)
    mg = mg0 + 0.3 * m_h + rng.normal(0, sig, n)
    mu = rng.uniform(9.0, 11.0, n)
    a0 = np.full(n, a0_true) if np.ndim(a0_true) == 0 else a0_true
    g = mg + mu + EXT_K["G"] * a0
    plx = 1000.0 / 10 ** ((mu + 5) / 5) + rng.normal(0, plx_err, n)
    return dict(g=g, plx=plx, err=np.full(n, plx_err), m_h=m_h, teff=teff, logg=logg)


def test_rc_selection_window():
    #               RC       RGB (logg high)  dwarf    hot subgiant  low logg   metal-poor  NaN
    teff = np.array([4750.0, 4750.0,          5700.0,  5300.0,       4750.0,    4750.0,     np.nan])
    logg = np.array([2.4,    3.0,             4.4,     2.4,          1.9,       2.4,        2.4])
    mh = np.array([-0.1,     -0.1,            0.0,     -0.1,         -0.1,      -1.2,       0.0])
    assert rc_select(teff, logg, mh).tolist() == [True, False, False, False, False, False, False]


def test_calibration_recovers_injected_mg_and_scatter():
    d = _fake_rc()
    cal = calibrate_rc(d["g"], d["plx"], d["err"], d["m_h"], d["teff"], d["logg"], np.zeros(len(d["g"])),
                       np.ones(len(d["g"])))
    assert cal["n"] > 3500 and cal["sigma_rc"] == pytest.approx(0.12, abs=0.015)
    mg = rc_mg(cal["coef"], np.array([-0.5, 0.2]), np.array([4800.0, 4600.0]))
    assert mg[1] - mg[0] > 0.1                                   # recovered [M/H] slope (0.3 per dex, dT-correlated)
    # high-extinction stars with the same truth -> targets recover the injected A0 (unbiased, within sigma)
    for a0_true in (0.0, 2.5):
        e = _fake_rc(seed=2, a0_true=a0_true)
        t, s, ok = rc_anchor_targets(e["g"], e["plx"], e["err"], e["m_h"], e["teff"], e["logg"], cal)
        assert ok.mean() > 0.8 and abs(np.median(t[ok]) - a0_true) < 0.05
        assert np.all(s[ok] >= np.hypot(cal["sigma_rc"], 0.1) / EXT_K["G"] - 1e-9)       # sigma floor


def test_targets_sigma_snr_cut_and_no_zero_clip():
    cal = {"coef": [0.55, 0.0, 0.0, 0.0], "sigma_rc": 0.12}
    g = np.array([11.0, 11.0, 11.0, 11.0])
    mu = 10.0                                                    # plx_corr = 1 mas -> mu = 10
    plx = np.array([1.0, 1.0, 1.0, -1.0])
    err = np.array([0.05, 0.4, 0.1, 0.05])                       # S/N 20, 2.5 (cut), 10, negative parallax
    teff, logg, mh = np.full(4, 4750.0), np.full(4, 2.4), np.full(4, -0.1)
    t, s, ok = rc_anchor_targets(g, plx, err, mh, teff, logg, cal)
    assert ok.tolist() == [True, False, True, False]
    assert t[0] == pytest.approx((11.0 - 0.55 - 10.0) / EXT_K["G"])
    # sigma_mu = 5/ln10 * err/plx: row 0 0.109 mag (just above the 0.1 floor), row 2 0.217 mag
    assert s[0] == pytest.approx(np.hypot(0.12, 5 / np.log(10) * 0.05) / EXT_K["G"])
    t3, s3, _ = rc_anchor_targets(g[:1], np.array([1.0]), np.array([0.01]), mh[:1], teff[:1], logg[:1], cal)
    assert s3[0] == pytest.approx(np.hypot(0.12, 0.1) / EXT_K["G"])             # floor
    assert s[2] == pytest.approx(np.hypot(0.12, 5 / np.log(10) * 0.1) / EXT_K["G"])
    # brighter than the clump at that parallax -> negative target is kept (not clipped to zero)
    t2, _, ok2 = rc_anchor_targets(np.array([10.0]), np.array([1.0]), np.array([0.05]), mh[:1], teff[:1], logg[:1], cal)
    assert ok2[0] and t2[0] < 0
    # the anchor loss uses (A0 - target)/sigma
    a0 = torch.tensor([0.5]); tg = torch.tensor([float(t[0])]); sg = torch.tensor([float(s[0])])
    assert float(anchor_nll(a0, tg, sg, torch.tensor([True]))) == pytest.approx(0.5 * ((0.5 - t[0]) / s[0]) ** 2)


def test_dataset_rc_anchors_train_only_and_disjoint(tmp_path):
    import pandas as pd
    _toy_root(tmp_path, with_aux=False)
    t = _toy_astro(tmp_path, n=60)
    rng = np.random.default_rng(0)
    n = len(t)
    e = _fake_rc(n=n, a0_true=1.0)
    t["parallax"], t["parallax_error"], t["phot_g_mean_mag"] = e["plx"] + 0.03, e["err"], e["g"]
    t["ebv_sfd"] = 0.01
    t["b"] = 5.0                                                 # low latitude: no high-latitude anchors
    t.to_parquet(tmp_path / "astrometry_phot.parquet")
    z = dict(np.load(tmp_path / "labels.npz"))
    from aspgapv3.data.dataset import LabelScaler
    names = LabelScaler.load(tmp_path).labels
    for k, v in (("teff", e["teff"]), ("logg", e["logg"]), ("m_h", e["m_h"])):
        z["y"][:, names.index(k)] = v
        z["mask"][:, names.index(k)] = True
    np.savez(tmp_path / "labels.npz", **z)
    cfg = {"enabled": True, "snr_min": 5.0, "sig_mu_floor": 0.1}
    tr = APOGEEDataset_(tmp_path, "train", cfg)
    va = APOGEEDataset_(tmp_path, "val", cfg)
    assert (tr.anchor_set == RC_SET).sum() > 20 and (va.anchor_set == 0).all()
    assert tr.rc_calib["sigma_rc"] > 0
    i = int(np.flatnonzero(tr.anchor_set == RC_SET)[0])
    a = tr[i]["astro"]
    assert bool(a["rc_ok"]) and not bool(a["anc_ok"]) and a["anc_s"] > 0.2
    assert abs(float(np.median(tr.astro_arr["anc_t"][tr.anchor_set == RC_SET]))) < 0.3   # self-calibrated on A_SFD~0.03 stars
    off = APOGEEDataset_(tmp_path, "train", None)
    assert not off.astro_arr["rc_ok"].any()


def APOGEEDataset_(root, split, rc):
    from aspgapv3.data.dataset import APOGEEDataset
    return APOGEEDataset(root, split, labels=["teff", "logg", "m_h", "alpha_m"], astro=True, rc_anchor=rc)
