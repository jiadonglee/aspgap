"""Side conditioning, attention pooling, precision policy, backward compat."""
from pathlib import Path

import pytest
import torch

from aspgapv3.amp import amp_settings, autocast_ctx, float_outputs
from aspgapv3.config import load_config, resolve_ckpt
from aspgapv3.losses import MaskedHeteroscedasticNLL
from aspgapv3.models.encoders import AttnPool, default_specclip_ckpt
from aspgapv3.models.net import XPChemNet, build_model
from aspgapv3.models.side import SideFiLM, build_side

CFG = Path(__file__).resolve().parents[1] / "configs"


def have_specclip():
    try:
        import specclip  # noqa: F401
    except ImportError:
        return False
    return default_specclip_ckpt() is not None


def scratch_net(side=None):
    cfg = {"model": {"encoder": {"type": "scratch", "d_model": 32, "n_layers": 1,
                                 "n_heads": 4, "embed_dim": 768},
                     "trunk": {"dim": 256, "dropout": 0.0}, "side": side}}
    return build_model(cfg, n_atm=3, n_chem=3)


def side_batch(B=4):
    return {"side": torch.randn(B, 4), "sigma": torch.rand(B, 343) * 0.05,
            "x": torch.rand(B, 343) + 0.5}


def test_film_zero_init_is_identity():
    torch.manual_seed(0)
    net = scratch_net({"scalars": ["g", "bp_rp", "log_flux550", "log_snr550"],
                       "noise_channel": True}).eval()
    b = side_batch()
    base = net(b["x"])                                   # side=None -> no FiLM
    with_side = net(b["x"], build_side(net, b))
    for k in base:
        torch.testing.assert_close(base[k], with_side[k], rtol=0, atol=0)
    # and it is not stuck at identity: perturb the output layer
    torch.nn.init.normal_(net.side.out.weight, std=0.1)
    moved = net(b["x"], build_side(net, b))
    assert (moved["z"] - base["z"]).abs().max() > 1e-4


def test_film_gradient_reaches_side_when_zero_init():
    net = scratch_net({"scalars": ["g"], "noise_channel": True})
    b = side_batch()
    net(b["x"], build_side(net, b))["mu_chem"].sum().backward()
    assert net.side.out.weight.grad.abs().sum() > 0     # out layer learns first


def test_scalar_stats_buffers_roundtrip():
    f = SideFiLM(256, ["g", "bp_rp"], False)
    f.set_scalar_stats([14.0, 1.0], [1.5, 0.5])
    f2 = SideFiLM(256, ["g", "bp_rp"], False)
    f2.load_state_dict(f.state_dict())
    assert f2.s_std.tolist() == [1.5, 0.5]


def test_attn_pool_shape_and_param_groups():
    pool = AttnPool(768, n_queries=4, n_heads=2)
    assert pool(torch.randn(2, 344, 768)).shape == (2, 768)
    net = scratch_net()
    net.encoder.pool = AttnPool(768, 2, 2)               # emulate SpecCLIP + attn
    ids = {id(p) for g in net.optimizer_param_groups(3e-4, 1e-5, 0.01)
           if g["name"] == "heads" for p in g["params"]}
    assert all(id(p) in ids for p in net.encoder.pool.parameters())


@pytest.mark.skipif(not have_specclip(), reason="SpecCLIP ckpt unavailable")
def test_specclip_attn_pooling_and_pool_lr_group():
    cfg = resolve_ckpt(load_config(CFG / "pixel_fusion.yaml"))
    net = build_model(cfg, n_atm=3, n_chem=3)
    out = net(torch.rand(2, 343) + 0.5, side_batch(2) and build_side(net, side_batch(2)))
    assert out["mu_chem"].shape == (2, 3) and torch.isfinite(out["z"]).all()
    groups = net.optimizer_param_groups(3e-4, 1e-5, 0.01)
    heads = {id(p) for g in groups if g["name"] == "heads" for p in g["params"]}
    assert all(id(p) in heads for p in net.encoder.pool.parameters())
    assert len(net.encoder_parameters()) > 0


@pytest.mark.parametrize("name", ["aspgap110", "scratch", "specclip_frozen", "specclip_ft"])
def test_old_configs_still_build_and_forward(name):
    cfg = resolve_ckpt(load_config(CFG / f"{name}.yaml"))
    if cfg["model"]["encoder"]["type"] == "specclip" and not have_specclip():
        pytest.skip("SpecCLIP ckpt unavailable")
    net = build_model(cfg, n_atm=3, n_chem=3).eval()
    assert net.side is None
    x = torch.rand(2, 110 if cfg["data"]["input"].startswith("coef") else 343) + 0.5
    out = net(x)
    assert out["mu_atm"].shape == (2, 3) and torch.isfinite(out["z"]).all()
    assert amp_settings(cfg, "cuda") == (True, torch.bfloat16)


def test_pixel_fusion_configs_parse():
    a = load_config(CFG / "pixel_fusion.yaml")
    b = load_config(CFG / "pixel_fusion_noaug.yaml")
    assert a["train"]["augment"]["noise_match"]["p"] == 0.5
    assert b["train"]["augment"]["noise_match"]["p"] == 0.0
    assert b["model"]["encoder"]["pooling"] == "attn" and b["train"]["lr_enc"] == 1e-5


def test_loss_is_fp32_under_bf16_autocast():
    net = scratch_net()
    with autocast_ctx(True, torch.bfloat16, "cpu"):
        out = net(torch.rand(4, 343) + 0.5)
        assert out["mu_chem"].dtype == torch.bfloat16     # forward really ran in bf16
    out = float_outputs(out)
    assert out["mu_chem"].dtype == torch.float32
    nll = MaskedHeteroscedasticNLL(torch.ones(3), torch.full((3,), 0.05))
    y = torch.randn(4, 3)
    loss = nll(out["mu_chem"], out["logvar_chem"], y, torch.zeros(4, 3),
               torch.ones(4, 3, dtype=torch.bool))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    # logvar that overflows bf16->exp stays finite in fp32 after the clamp
    big = torch.full((4, 3), 1e4)
    assert torch.isfinite(nll(out["mu_chem"], big, y, torch.zeros(4, 3),
                              torch.ones(4, 3, dtype=torch.bool)))


def test_model_scalar_selection_does_not_affect_augmentation():
    from test_augment import make_bank, make_batch
    from test_xpnoise import synth_dm
    from aspgapv3.augment import NoiseMatchAugment
    dm = synth_dm()
    aug = NoiseMatchAugment(dm, make_bank(), device="cpu", p=1.0)
    batch, *_ = make_batch(dm)
    nets = [scratch_net({"scalars": ["log_flux550", "log_snr550"], "noise_channel": True}),
            scratch_net({"scalars": ["g", "bp_rp"], "noise_channel": False})]
    outs = []
    for net in nets:                                   # augmentation never sees the net
        build_side(net, batch)
        outs.append(aug(dict(batch), gen=torch.Generator().manual_seed(3), fixed_g=16.0))
    torch.testing.assert_close(outs[0]["x"], outs[1]["x"], rtol=0, atol=0)
    assert (outs[0]["side"][:, 0] == 16.0).all()       # G (col 0) was used for s
    assert nets[0].side.cols == [2, 3] and nets[1].side.cols == [0, 1]
    assert build_side(nets[0], outs[0])["scalars"].shape == (len(batch["x"]), 2)


def test_pixel_fusion_phot_config():
    c = load_config(CFG / "pixel_fusion_phot.yaml")
    assert c["model"]["side"]["scalars"] == ["g", "bp_rp", "log_flux550", "log_snr550"]
    assert load_config(CFG / "pixel_fusion.yaml")["model"]["side"]["scalars"] == \
        ["log_flux550", "log_snr550"]
    assert load_config(CFG / "base.yaml")["eval"]["amp"] is False
