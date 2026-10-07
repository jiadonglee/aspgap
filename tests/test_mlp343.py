"""mlp343_aug building blocks: in_dim generalisation, in-network input standardisation."""
import numpy as np
import pytest
import torch

from aspgapv3.augment import NoiseMatchAugment
from aspgapv3.config import load_config
from aspgapv3.models.encoders import CoeffMLPEncoder
from aspgapv3.models.net import build_model

from test_augment import make_bank, make_batch
from test_select_decoupled import run_train
from test_xpnoise import synth_dm

from pathlib import Path
CFG = Path(__file__).resolve().parents[1] / "configs"


def mlp_cfg(in_norm=None, in_dim=343):
    return {"model": {"encoder": {"type": "coeff_mlp", "in_dim": in_dim, "embed_dim": 768,
                                  "unfreeze_last_k": -1},
                      "trunk": {"dim": 64, "dropout": 0.0},
                      **({"input_norm": in_norm} if in_norm else {})}}


def test_coeff_mlp_old_state_dict_unchanged_and_in_dim_generalises():
    old = CoeffMLPEncoder()
    assert [(k, tuple(v.shape)) for k, v in old.state_dict().items()][0] == ("net.0.weight", (512, 110))
    assert list(old.state_dict()) == list(CoeffMLPEncoder(in_dim=343).state_dict())
    assert CoeffMLPEncoder(in_dim=343).state_dict()["net.0.weight"].shape == (512, 343)
    net = build_model({"model": {"encoder": {"type": "coeff_mlp", "embed_dim": 768, "unfreeze_last_k": -1},
                                 "trunk": {"dim": 64, "dropout": 0.0}}}, 3, 3)
    assert not net.input_norm and not any("in_" in k for k in net.state_dict())   # old models: no new buffers
    net.load_state_dict(net.state_dict())


def test_input_standardisation_roundtrip_and_equals_manual():
    torch.manual_seed(0)
    a = build_model(mlp_cfg("robust"), 3, 3).eval()
    b = build_model(mlp_cfg(None), 3, 3).eval()
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith("in_")})
    x = torch.rand(6, 343) * 2 + .5
    c, s = x.median(0).values, (x.quantile(.75, 0) - x.quantile(.25, 0))
    a.set_input_stats(c, s)
    out_a, out_b = a(x), b((x - c) / s.clamp_min(1e-4))
    for k in out_a:
        torch.testing.assert_close(out_a[k], out_b[k])
    z = (x - a.in_center) / a.in_scale                                  # round trip of the stats
    torch.testing.assert_close(z * a.in_scale + a.in_center, x)
    # stats travel in the state dict
    a2 = build_model(mlp_cfg("robust"), 3, 3)
    a2.load_state_dict(a.state_dict())
    torch.testing.assert_close(a2.in_center, a.in_center)
    with pytest.raises(ValueError, match="input_norm"):
        build_model(mlp_cfg("zscore"), 3, 3)


def test_augmented_inputs_get_same_standardisation():
    dm = synth_dm()
    batch, _, _ = make_batch(dm, n=5, g=13.0)
    aug = NoiseMatchAugment(dm, make_bank(), device="cpu", p=1.0)
    out = aug(batch, gen=torch.Generator().manual_seed(0), fixed_g=17.0)
    assert out["aug"].any() and not torch.equal(out["x"], batch["x"])
    net = build_model(mlp_cfg("robust"), 3, 3).eval()
    c, s = batch["x"].median(0).values, batch["x"].std(0) + .1
    net.set_input_stats(c, s)
    ref = build_model(mlp_cfg(None), 3, 3).eval()
    ref.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("in_")})
    torch.testing.assert_close(net(out["x"])["mu_chem"], ref((out["x"] - c) / s)["mu_chem"])


def test_new_configs_settings():
    c = lambda n: load_config(CFG / f"{n}.yaml")
    m = c("mlp343_aug")
    assert m["data"]["input"] == "xp343" and m["model"]["input_norm"] == "robust"
    assert m["model"]["encoder"]["type"] == "coeff_mlp" and m["model"]["encoder"]["in_dim"] == 343
    assert not m["model"].get("side") and m["train"]["lr_enc"] == 3e-4 and m["train"]["epochs"] == 60
    assert m["train"]["select_on"] == "mean" and m["train"]["augment"]["noise_match"]["p"] == 0.5
    for n, base in (("aspgap110_aug_p1", "aspgap110_aug"), ("mlp343_aug_p1", "mlp343_aug")):
        p = c(n)
        assert p["labels"]["aux"] == ["c_h", "n_h", "o_h", "si_h", "ca_h", "ti_h", "al_h", "mn_h", "ni_h"]
        assert p["labels"]["aux_ref"] == "fe" and p["labels"]["weight_aux"] == 0.3
        assert p["model"]["heads"]["chem_condition"] == "atm_detached"
        assert p["data"]["input"] == c(base)["data"]["input"]
    assert c("aspgap110_aug")["model"].get("input_norm") in (None, "none")   # unchanged


def test_train_computes_input_stats_and_saves_them(tmp_path, monkeypatch):
    model = {"input_norm": "robust",
             "encoder": {"type": "coeff_mlp", "in_dim": 343, "embed_dim": 768, "unfreeze_last_k": -1},
             "trunk": {"dim": 64, "dropout": 0.0}}
    rows, run = run_train(tmp_path, monkeypatch,
                          {"select_on": "mean", "augment": {"noise_match": {"p": 0.5, "g_max": 17.65}}},
                          model=model, ckpt=True)
    assert any(r["event"] == "input_norm" for r in rows)
    sd = torch.load(run / "ckpt_best.pt", weights_only=False)["state_dict"]
    assert sd["in_center"].shape == (343,) and (sd["in_scale"] > 0).all()
    assert not torch.equal(sd["in_scale"], torch.ones(343))
