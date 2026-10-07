"""Latent-distillation head + random-sampler ablation: default off, shapes, alignment, loss."""
from pathlib import Path

import numpy as np
import pytest
import torch

from aspgapv3.config import load_config
from aspgapv3.data.dataset import APOGEEDataset
from aspgapv3.data.sampler import make_sampler
from aspgapv3.losses import combine_losses, masked_latent_mse
from aspgapv3.models.net import build_model
from test_p1 import _toy_root, scratch_cfg

CFG = Path(__file__).resolve().parents[1] / "configs"
LAT = {"source": "v1_latent", "dim": 16, "hidden": 8, "weight": 1.0}


def _cfg(latent=None):
    c = scratch_cfg()
    if latent is not None:
        c["model"]["latent_head"] = latent
    return c


def test_default_off_everywhere():
    # no config except aspgap110_aug_v1lat declares a latent head
    for p in sorted(CFG.glob("*.yaml")):
        has = bool((load_config(p, []).get("model") or {}).get("latent_head"))
        assert has == (p.stem == "aspgap110_aug_v1lat"), p.stem
    # model: identical parameter set / forward with and without the (absent or empty) config
    torch.manual_seed(0)
    a = build_model(_cfg(), n_atm=3, n_chem=3).eval()
    torch.manual_seed(0)
    b = build_model(_cfg({}), n_atm=3, n_chem=3).eval()
    assert a.head_lat is None and b.head_lat is None
    assert list(a.state_dict()) == list(b.state_dict())
    assert not any("head_lat" in k for k in a.state_dict())
    x = torch.rand(4, 343) + .5
    oa, ob = a(x), b(x)
    assert "lat" not in oa and set(oa) == set(ob)
    assert all(torch.equal(oa[k], ob[k]) for k in oa)
    # old state dict loads strictly into a default model
    b.load_state_dict(a.state_dict())


def test_combine_losses_unchanged_without_latent():
    la, lc, lx = torch.tensor(.3), torch.tensor(.5), torch.tensor(.2)
    for mode in ("per_group", "global"):
        old = combine_losses(la, lc, lx, mode, .5, .3)
        assert torch.equal(combine_losses(la, lc, lx, mode, .5, .3, l_lat=None, weight_lat=7.0), old)
        assert torch.equal(combine_losses(la, lc, None, mode, .5, .3, None, 7.0),
                           combine_losses(la, lc, None, mode, .5, .3))
        assert torch.allclose(combine_losses(la, lc, lx, mode, .5, .3, torch.tensor(.4), 2.0), old + .8)


def test_latent_head_shape_params_and_gradient():
    net = build_model(_cfg(LAT), n_atm=3, n_chem=3)
    out = net(torch.rand(4, 343) + .5)
    assert out["lat"].shape == (4, 16)
    assert any(k.startswith("head_lat") for k in net.state_dict())
    assert {id(p) for p in net.head_lat.parameters()} <= {id(p) for p in net.head_parameters()}
    masked_latent_mse(out["lat"], torch.randn(4, 16)).backward()
    assert net.trunk.net[0].weight.grad.abs().sum() > 0               # distillation shapes the trunk
    assert build_model(_cfg({"source": None, "dim": 16}), 3, 3).head_lat is None
    lin = build_model(_cfg({**LAT, "hidden": 0}), 3, 3)
    assert isinstance(lin.head_lat, torch.nn.Linear)


def test_masked_latent_mse_is_fp32_and_masked():
    p = torch.randn(5, 8, dtype=torch.bfloat16)
    t = torch.randn(5, 8)
    ok = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
    l = masked_latent_mse(p, t, ok)
    assert l.dtype == torch.float32
    ref = ((p.float() - t) ** 2).mean(1)[ok].mean()
    torch.testing.assert_close(l, ref)


def _write_latent(root, n=60, d=16, sid_shift=0):
    rng = np.random.default_rng(1)
    lat = (rng.normal(size=(n, d)) * 0.02 + 3.0).astype(np.float32)
    tr = np.arange(n) < 40
    np.save(root / "v1_latent.npy", lat)
    np.savez(root / "v1_latent_norm.npz", mean=lat[tr].mean(0), std=lat[tr].std(0),
             source_id=np.arange(100, 100 + n) + sid_shift, valid=np.ones(n, bool))
    return lat


def test_dataset_latent_standardized_aligned_and_optional(tmp_path):
    _toy_root(tmp_path, with_aux=False)
    lat = _write_latent(tmp_path)
    assert "lat" not in APOGEEDataset(tmp_path, "train")[0]           # default: no latent key
    ds = APOGEEDataset(tmp_path, "val", latent="v1_latent")
    it = ds[3]
    row = ds.indices[3]
    assert it["lat"].shape == (16,) and bool(it["lat_ok"])
    np.testing.assert_allclose(it["lat"].numpy() * ds.lat_std + ds.lat_mean, lat[row], rtol=1e-5, atol=1e-5)
    tr = APOGEEDataset(tmp_path, "train", latent="v1_latent")          # train stats => ~N(0,1)
    Z = np.stack([tr[i]["lat"].numpy() for i in range(len(tr))])
    assert np.abs(Z.mean(0)).max() < 1e-3 and np.abs(Z.std(0) - 1).max() < 1e-3
    with pytest.raises(ValueError, match="source"):
        APOGEEDataset(tmp_path, "train", latent="nope")


def test_dataset_latent_misalignment_and_missing(tmp_path):
    _toy_root(tmp_path, with_aux=False)
    with pytest.raises(FileNotFoundError, match="build_v1_latent"):
        APOGEEDataset(tmp_path, "train", latent="v1_latent")
    _write_latent(tmp_path, sid_shift=1)
    with pytest.raises(ValueError, match="aligned"):
        APOGEEDataset(tmp_path, "train", latent="v1_latent")


def test_random_sampler_config_and_distillation_config():
    base = load_config(CFG / "aspgap110_aug.yaml", [])
    rnd = load_config(CFG / "aspgap110_aug_rand.yaml", [])
    assert rnd["train"]["sampler"]["type"] == "random" and make_sampler(None, rnd["train"]["sampler"]) is None
    base["run"]["name"], base["train"]["sampler"]["type"] = rnd["run"]["name"], "random"
    assert base == rnd                                                 # only the sampler differs
    v = load_config(CFG / "aspgap110_aug_v1lat.yaml", [])
    lh = v.pop("model")["latent_head"]
    assert lh["source"] == "v1_latent" and lh["dim"] == 1024 and lh["weight"] > 0
    ref = load_config(CFG / "aspgap110_aug.yaml", [])
    ref.pop("model")
    v["run"] = ref["run"] = None
    assert v == ref                                                    # everything else identical
