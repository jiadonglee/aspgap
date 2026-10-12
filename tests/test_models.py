import torch

from aspgapv3.models.encoders import CoeffMLPEncoder
from aspgapv3.models.heads import HeteroscedasticHead, Trunk
from aspgapv3.models.net import XPChemNet, build_model
from toy import tiny_cfg


def test_coeff_mlp_encoder_shapes():
    enc = CoeffMLPEncoder(hidden=64)
    assert enc(torch.randn(5, 110)).shape == (5, 768)


def test_trunk_and_head_shapes():
    t = Trunk(768, 256)
    z = t(torch.randn(4, 768))
    assert z.shape == (4, 256)
    mu, lv = HeteroscedasticHead(256, 3)(z)
    assert mu.shape == lv.shape == (4, 3)


def make_net():
    return XPChemNet(CoeffMLPEncoder(hidden=64), trunk_dim=256, n_atm=3, n_chem=1)


def test_net_forward_dict():
    out = make_net()(torch.randn(6, 110))
    for key in ("z768", "z", "mu_atm", "logvar_atm", "mu_chem", "logvar_chem"):
        assert key in out
    assert out["mu_atm"].shape == (6, 3) and out["mu_chem"].shape == (6, 1)
    assert "mg" not in out                          # no distance head unless enabled


def test_param_groups_split_encoder_from_heads():
    net = make_net()
    groups = net.optimizer_param_groups(lr_head=3e-4, lr_enc=1e-5, weight_decay=0.01)
    assert {g["name"] for g in groups} == {"heads", "encoder"}
    enc_ids = {id(p) for p in net.encoder.parameters()}
    head_ids = {id(p) for g in groups if g["name"] == "heads" for p in g["params"]}
    assert not (enc_ids & head_ids)
    assert sum(len(g["params"]) for g in groups) == len(list(net.parameters()))
    lr = {g["name"]: g["lr"] for g in groups}
    assert lr["encoder"] == 1e-5 and lr["heads"] == 3e-4
    assert {g["name"] for g in net.optimizer_param_groups(3e-4, 0.0, 0.01)} == {"heads"}


def test_build_model_from_cfg_and_state_dict_keys():
    net = build_model(tiny_cfg(distance=True), n_atm=3, n_chem=1)
    assert net(torch.randn(2, 110))["mu_chem"].shape == (2, 1)
    keys = list(net.state_dict())
    assert {k.split(".")[0] for k in keys} == {"encoder", "trunk", "head_atm", "head_chem", "head_dist", "colour_net"}
