import numpy as np
import pytest
import torch

from aspgapv3.models.encoders import (CoeffMLPEncoder, ScratchXPTransformer,
                                      build_encoder)
from aspgapv3.models.heads import HeteroscedasticHead, Trunk
from aspgapv3.models.net import XPChemNet, build_model


def test_scratch_encoder_shapes():
    enc = ScratchXPTransformer(d_model=64, n_layers=2, n_heads=4)
    x = torch.randn(3, 343)
    z = enc(x)
    assert z.shape == (3, 768)
    assert torch.isfinite(z).all()
    assert all(p.requires_grad for p in enc.parameters())


def test_coeff_mlp_encoder_shapes():
    enc = CoeffMLPEncoder(hidden=64)
    z = enc(torch.randn(5, 110))
    assert z.shape == (5, 768)


def test_build_encoder_unknown_type():
    with pytest.raises(ValueError):
        build_encoder({"type": "nope"})


def test_trunk_and_head_shapes():
    t = Trunk(768, 256)
    z = t(torch.randn(4, 768))
    assert z.shape == (4, 256)
    h = HeteroscedasticHead(256, 3)
    mu, lv = h(z)
    assert mu.shape == lv.shape == (4, 3)


def make_net(frozen=True):
    enc = ScratchXPTransformer(d_model=64, n_layers=2, n_heads=4)
    if frozen:
        for p in enc.parameters():
            p.requires_grad_(False)
    return XPChemNet(enc, trunk_dim=256, n_atm=3, n_chem=3)


def test_net_forward_dict():
    net = make_net()
    out = net(torch.randn(6, 343))
    for key in ("z768", "z", "mu_atm", "logvar_atm", "mu_chem", "logvar_chem"):
        assert key in out
    assert out["mu_atm"].shape == (6, 3)
    assert out["mu_chem"].shape == (6, 3)


def test_frozen_encoder_has_no_trainable_params():
    net = make_net(frozen=True)
    assert len(net.encoder_parameters()) == 0
    # the SpecCLIP wrapper additionally pins eval() mode; that behavior is
    # covered in test_specclip_encoder.py against the real checkpoint


def test_param_groups_split_encoder_from_heads():
    net = make_net(frozen=False)
    groups = net.optimizer_param_groups(lr_head=3e-4, lr_enc=1e-5,
                                        weight_decay=0.01)
    names = {g["name"] for g in groups}
    assert names == {"heads", "encoder"}
    enc_ids = {id(p) for p in net.encoder_parameters()}
    head_params = {id(p) for g in groups if g["name"] == "heads" for p in g["params"]}
    assert not (enc_ids & head_params)
    lr = {g["name"]: g["lr"] for g in groups}
    assert lr["encoder"] == 1e-5 and lr["heads"] == 3e-4


def test_build_model_from_cfg():
    cfg = {"model": {"encoder": {"type": "scratch", "d_model": 64, "n_layers": 2,
                                 "n_heads": 4, "embed_dim": 768},
                     "trunk": {"dim": 256, "dropout": 0.1}}}
    net = build_model(cfg, n_atm=3, n_chem=3)
    assert net(torch.randn(2, 343))["mu_chem"].shape == (2, 3)
