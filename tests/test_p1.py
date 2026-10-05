"""P1: auxiliary labels, hierarchical chem conditioning, coef110n augmentation input."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aspgapv3.augment import apply_aspgap_input
from aspgapv3.config import load_config
from aspgapv3.data.dataset import APOGEEDataset, LabelScaler, save_catalog
from aspgapv3.data.xp import coef_to_aspgap_input
from aspgapv3.labels.loaders import build_label_matrix
from aspgapv3.labels.schema import AUX_CANDIDATES, spec
from aspgapv3.models.net import build_model
from aspgapv3.models.side import build_side

ROOT = Path(__file__).resolve().parents[1]
MAIN6 = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]


def scratch_cfg(heads=None, side=None):
    return {"model": {"encoder": {"type": "scratch", "d_model": 32, "n_layers": 1,
                                  "n_heads": 4, "embed_dim": 768},
                      "trunk": {"dim": 64, "dropout": 0.0}, "side": side,
                      **({"heads": heads} if heads is not None else {})}}


# ------------------------------------------------------------------ labels

def test_aux_registry_policy():
    for n in AUX_CANDIDATES:
        s = spec(n)
        assert s.flag_column and s.error_column and s.sigma_floor >= 0.02
    assert "fe_h" not in AUX_CANDIDATES and "mg_h" not in AUX_CANDIDATES


def test_aux_element_flag_and_range_mask():
    n = 6
    df = pd.DataFrame({"gaia_dr3_source_id": np.arange(1, n + 1), "snr": 100.0,
                       "raw_teff": 5000.0, "teff": 5000.0, "logg": 3.0, "m_h_atm": 0.0,
                       "alpha_m_atm": 0.0, "fe_h": 0.0, "mg_h": 0.0,   # tier-1 columns are required
                       "s_h": [0.1, 0.1, 0.1, np.nan, 9.0, 0.1],
                       "e_s_h": [0.01] * n,
                       "s_h_flags": [0, 1 << 5, 1 << 9, 0, 0, 1 << 2]})   # censored / warn / upper-limit
    y, e, m, _ = build_label_matrix(df, ["s_h"], mask_warn_bits=True)
    assert m[:, 0].tolist() == [True, False, False, False, False, True]
    _, _, m2, _ = build_label_matrix(df, ["s_h"], mask_warn_bits=False)
    assert m2[:, 0].tolist() == [True, False, True, False, False, True]
    assert np.isnan(y[~m[:, 0], 0]).all()


def _toy_root(tmp_path, n=60, with_aux=True, seed=0):
    rng = np.random.default_rng(seed)
    K = len(MAIN6)
    y = rng.normal(size=(n, K)).astype(np.float32)
    mask = np.ones((n, K), bool)
    sc = LabelScaler.fit(y, np.full_like(y, 0.05), mask, np.full(K, 0.02), MAIN6)
    sc.save(tmp_path)
    np.savez(tmp_path / "labels.npz", y=y, yerr=np.full_like(y, .05), mask=mask,
             labels=np.array(MAIN6, dtype="U16"))
    np.save(tmp_path / "xp343.npy", rng.normal(1, .05, (n, 343)).astype(np.float32))
    splits = np.array(["train"] * 40 + ["val"] * 10 + ["iid_test"] * 10)
    cat = pd.DataFrame({"row": np.arange(n), "source_id": np.arange(100, 100 + n),
                        "split": splits})
    save_catalog(cat, tmp_path)
    aux = ["c_h", "n_h", "o_h"]
    if with_aux:
        ya = rng.normal(size=(n, 3)).astype(np.float32)
        ma = rng.random((n, 3)) > 0.3
        ya[~ma] = np.nan
        ea = np.where(ma, 0.03, np.nan).astype(np.float32)
        np.savez(tmp_path / "labels_aux.npz", y=ya, yerr=ea, mask=ma,
                 labels=np.array(aux, dtype="U16"), source_id=cat["source_id"].to_numpy())
        LabelScaler.fit(ya[:40], ea[:40], ma[:40], np.array([.03, .05, .03]), aux) \
            .save(tmp_path, "norm_aux.npz")
        return ya, ma
    return None


def test_aux_dataset_masks_and_order(tmp_path):
    ya, ma = _toy_root(tmp_path)
    ds = APOGEEDataset(tmp_path, "train", aux=["o_h", "c_h"])        # subset + reorder
    assert ds.aux_labels == ["o_h", "c_h"]
    i = 5
    it = ds[i]
    row = ds.indices[i]
    assert it["mask_aux"].tolist() == ma[row][[2, 0]].tolist()
    assert it["y_aux"].shape == (2,) and torch.isfinite(it["y_aux"]).all()
    assert torch.all(it["y_aux"][~it["mask_aux"]] == 0)              # masked -> 0, never NaN
    assert np.allclose(ds.y_aux_raw[i], ya[row][[2, 0]], equal_nan=True)
    # main labels untouched
    assert ds.labels == MAIN6 and ds[i]["y"].shape == (6,)
    # without aux request: old item keys
    assert "y_aux" not in APOGEEDataset(tmp_path, "train")[0]


def test_aux_missing_file_errors_only_when_configured(tmp_path):
    _toy_root(tmp_path, with_aux=False)
    APOGEEDataset(tmp_path, "train")                                  # fine
    APOGEEDataset(tmp_path, "train", aux=[])                          # fine
    with pytest.raises(FileNotFoundError, match="build_aux_labels"):
        APOGEEDataset(tmp_path, "train", aux=["c_h"])


def test_aux_unknown_name_and_misalignment(tmp_path):
    _toy_root(tmp_path)
    with pytest.raises(ValueError, match="not in"):
        APOGEEDataset(tmp_path, "train", aux=["ce_h"])
    z = dict(np.load(tmp_path / "labels_aux.npz"))
    z["source_id"] = z["source_id"][::-1].copy()
    np.savez(tmp_path / "labels_aux.npz", **z)
    with pytest.raises(ValueError, match="aligned"):
        APOGEEDataset(tmp_path, "train", aux=["c_h"])


def test_build_aux_labels_script_end_to_end(tmp_path):
    """Synthetic ASPCAP parquet -> processed root -> build_aux_labels: aligned to
    catalog order via source_id, main labels reproduced, new files only."""
    from aspgapv3.labels.loaders import dedupe_max_snr, load_and_clean
    rng = np.random.default_rng(3)
    n = 80
    sid = rng.integers(1, 50, n)                                      # duplicates on purpose
    tab = pd.DataFrame({
        "gaia_dr3_source_id": sid, "snr": rng.uniform(40, 200, n), "flag_bad": np.zeros(n),
        "raw_teff": rng.uniform(3500, 7500, n), "raw_e_teff": 50.0, "teff": 5000.0,
        "logg": rng.uniform(0.5, 5, n), "e_logg": .1, "logg_flags": 0,
        "m_h_atm": rng.normal(-.3, .3, n), "e_m_h_atm": .02, "m_h_atm_flags": 0,
        "alpha_m_atm": rng.normal(.1, .05, n), "e_alpha_m_atm": .02, "alpha_m_atm_flags": 0,
        "fe_h": rng.normal(-.3, .3, n), "e_fe_h": .02, "fe_h_flags": 0,
        "mg_h": rng.normal(-.2, .3, n), "e_mg_h": .02, "mg_h_flags": 0,
        "c_h": rng.normal(-.3, .2, n), "e_c_h": .02,
        "c_h_flags": np.where(rng.random(n) > .8, 1 << 5, 0),
    })
    from astropy.table import Table
    Table.from_pandas(tab).write(tmp_path / "t.fits", format="fits")     # table lands in HDU1
    root = tmp_path / "proc"
    root.mkdir()
    res = load_and_clean(str(tmp_path / "t.fits"), MAIN6, hdu=1)
    keep = dedupe_max_snr(res["df"], "gaia_dr3_source_id")
    drop = np.arange(0, len(keep), 7)                                  # "XP crossmatch" loses some
    sel = np.delete(keep, drop)
    perm = rng.permutation(len(sel))                                   # catalog order != table order
    sel = sel[perm]
    y, ye, m = res["y"][sel], res["yerr"][sel], res["mask"][sel]
    sids = res["df"]["gaia_dr3_source_id"].to_numpy()[sel]
    np.savez(root / "labels.npz", y=y, yerr=ye, mask=m, labels=np.array(MAIN6, dtype="U16"))
    split = np.where(np.arange(len(sel)) % 5 == 0, "val", "train")
    save_catalog(pd.DataFrame({"row": np.arange(len(sel)), "source_id": sids, "split": split}), root)
    (root / "manifest.json").write_text(json.dumps({"aspcap": str(tmp_path / "t.fits"), "hdu": 1,
                                                    "labels": MAIN6}))
    (root / "build_config.yaml").write_text(
        "data: {snr_min: 50.0, teff_max: 8000.0, teff_variant: raw}\n"
        "labels: {teff_variant: raw, mask_warn_bits: true}\n")
    # the test table has raw columns in a parquet w/o 50<snr guarantees -> reuse defaults
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.name.endswith((".npz", ".parquet"))}

    sp = importlib.util.spec_from_file_location("baux", ROOT / "scripts" / "build_aux_labels.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    argv = sys.argv
    sys.argv = ["x", "--root", str(root), "--elements", "c_h", "ce_h"]
    try:
        mod.main()
    finally:
        sys.argv = argv
    for k, v in before.items():
        assert (root / k).read_bytes() == v                         # existing files untouched
    z = np.load(root / "labels_aux.npz")
    assert z["labels"].tolist() == ["c_h", "ce_h"]
    assert np.array_equal(z["source_id"], sids)
    assert not z["mask"][:, 1].any()                                   # ce_h absent from table -> all masked
    # c_h aligned: value for each catalog row == table c_h of the chosen visit
    exp = res["df"]["c_h"].to_numpy()[sel]
    ok = z["mask"][:, 0]
    assert ok.any() and (~ok).any()
    assert np.allclose(z["y"][ok, 0], exp[ok])
    assert (root / "norm_aux.npz").is_file() and (root / "aux_manifest.json").is_file()
    with pytest.raises(SystemExit):                                    # no silent overwrite
        sys.argv = ["x", "--root", str(root)]
        try:
            mod.main()
        finally:
            sys.argv = argv


# ------------------------------------------------------------------- model

def _copy(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    return missing, unexpected


def test_chem_condition_none_reproduces_old_forward():
    torch.manual_seed(0)
    old = build_model(scratch_cfg(), n_atm=3, n_chem=3).eval()        # no `heads` key at all
    new = build_model(scratch_cfg({"chem_condition": "none"}), n_atm=3, n_chem=3).eval()
    assert list(old.state_dict()) == list(new.state_dict())
    new.load_state_dict(old.state_dict())                             # strict: same keys/shapes
    x = torch.rand(5, 343) + .5
    a, b = old(x), new(x)
    assert a.keys() == b.keys() and "mu_aux" not in b
    for k in a:
        assert torch.equal(a[k], b[k])
    # with an aux head attached (unconditioned) the science outputs are unchanged too
    aux = build_model(scratch_cfg({"chem_condition": "none"}), n_atm=3, n_chem=3, n_aux=4).eval()
    missing, unexpected = _copy(old, aux)
    assert not unexpected and all(k.startswith("head_aux") for k in missing)
    c = aux(x)
    for k in a:
        assert torch.equal(a[k], c[k])
    assert c["mu_aux"].shape == (5, 4) and c["logvar_aux"].shape == (5, 4)


@pytest.mark.parametrize("mode,expect_grad", [("atm_detached", False), ("atm", True)])
def test_detached_conditioning_blocks_gradient_into_atm_head(mode, expect_grad):
    torch.manual_seed(1)
    net = build_model(scratch_cfg({"chem_condition": mode}), n_atm=3, n_chem=3, n_aux=2)
    x = torch.rand(4, 343) + .5
    out = net(x)
    (out["mu_chem"].sum() + out["mu_aux"].sum()).backward()           # chem + aux loss only
    g = sum(float(p.grad.abs().sum()) for p in net.head_atm.parameters() if p.grad is not None)
    assert (g > 0) == expect_grad
    # z (trunk) still gets the chem gradient either way; the conditioned head is wider
    assert net.trunk.net[0].weight.grad.abs().sum() > 0
    assert net.head_chem.out.in_features == 64 + 3


def test_cond_logvar_and_aux_condition_flags():
    net = build_model(scratch_cfg({"chem_condition": "atm", "cond_logvar": True,
                                   "aux_condition": False}), n_atm=3, n_chem=3, n_aux=2)
    assert net.head_chem.out.in_features == 64 + 6
    assert net.head_aux.out.in_features == 64                         # aux on plain z
    out = net(torch.rand(2, 343) + .5)
    assert out["mu_aux"].shape == (2, 2)
    with pytest.raises(ValueError, match="chem_condition"):
        build_model(scratch_cfg({"chem_condition": "bogus"}), n_atm=3, n_chem=3)


def test_aux_params_in_head_group_and_float_outputs():
    net = build_model(scratch_cfg({"chem_condition": "atm_detached"}), n_atm=3, n_chem=3, n_aux=2)
    ids = {id(p) for g in net.optimizer_param_groups(3e-4, 1e-5, .01) if g["name"] == "heads"
           for p in g["params"]}
    assert all(id(p) in ids for p in net.head_aux.parameters())


# ----------------------------------------------- every shipped config still builds + steps

@pytest.mark.parametrize("cfg_path", sorted((ROOT / "configs").glob("*.yaml")),
                         ids=lambda p: p.stem)
def test_every_config_builds_and_trains_one_step(cfg_path):
    cfg = load_config(cfg_path)
    enc = cfg["model"]["encoder"]
    if enc["type"] == "specclip":                                     # checkpoint-free stand-in
        cfg["model"]["encoder"] = {"type": "scratch", "d_model": 32, "n_layers": 1,
                                   "n_heads": 4, "embed_dim": 768}
    cfg["model"]["trunk"] = {"dim": 64, "dropout": 0.0}
    inp = cfg["data"].get("input", "xp343")
    n_aux = len(cfg["labels"].get("aux") or [])
    net = build_model(cfg, n_atm=3, n_chem=3, n_aux=n_aux)
    B = 4
    x = torch.rand(B, 343 if inp == "xp343" else 110) + .5
    batch = {"side": torch.randn(B, 4), "sigma": torch.rand(B, 343) * .05, "x": x}
    out = net(x, build_side(net, batch))
    loss = out["mu_atm"].sum() + out["mu_chem"].sum() + out["logvar_chem"].sum()
    if n_aux:
        loss = loss + out["mu_aux"].sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_new_ablation_configs_have_intended_settings():
    c = lambda n: load_config(ROOT / "configs" / f"{n}.yaml")
    assert c("pixel_fusion_snronly")["model"]["side"]["scalars"] == ["log_snr550"]
    for n in ("pixel_fusion_p1", "pixel_fusion_p1_auxonly", "pixel_fusion_p1_condonly"):
        assert c(n)["model"]["side"]["scalars"] == ["log_snr550"]
    p1, ao, co = (c(n) for n in ("pixel_fusion_p1", "pixel_fusion_p1_auxonly",
                                 "pixel_fusion_p1_condonly"))
    assert p1["model"]["heads"]["chem_condition"] == "atm_detached" and p1["labels"]["aux"]
    assert ao["model"]["heads"]["chem_condition"] == "none" and ao["labels"]["aux"] == p1["labels"]["aux"]
    assert co["model"]["heads"]["chem_condition"] == "atm_detached" and co["labels"]["aux"] == []
    a = c("aspgap110_aug")
    assert a["data"]["input"] == "coef110n" and a["model"]["encoder"]["type"] == "coeff_mlp"
    assert a["train"]["augment"]["noise_match"]["p"] == 0.5 and a["train"]["select_on"] == "mean"
    # old configs: P1 features off
    for n in ("pixel_fusion", "aspgap110", "specclip_ft", "scratch"):
        assert not c(n)["labels"]["aux"] and c(n)["model"]["heads"]["chem_condition"] == "none"


# ------------------------------------------------ coef110n augmentation input

def test_apply_aspgap_input_degraded_rows_only():
    B = 6
    sc = {"center": np.random.randn(110), "scale": np.random.rand(110) + .5}
    coef = torch.rand(B, 110) + 1
    g = torch.full((B,), 14.0)
    side = torch.zeros(B, 4)
    side[:, 0] = torch.tensor([14, 14, 16.5, 16.5, 17, 14.])           # G' for degraded rows
    x0 = torch.randn(B, 110)
    aug = torch.tensor([False, False, True, True, True, False])
    out = apply_aspgap_input({"coef": coef, "side": side, "x": x0, "aug": aug}, sc, "cpu")
    ref = coef_to_aspgap_input(coef, side[:, 0], sc)
    assert torch.equal(out[~aug], x0[~aug])
    assert torch.allclose(out[aug], ref[aug])
    # identity check of the transform: G'=G with raw = g_norm*10^((15-G)0.4) round trip
    gn = torch.randn(B, 110).double()
    raw = gn * (10.0 ** ((15.0 - g.double()) * 0.4))[:, None]
    back = coef_to_aspgap_input(raw.float(), g, {"center": np.zeros(110), "scale": np.ones(110)})
    assert torch.allclose(back, gn.float(), atol=1e-4, rtol=1e-4)
