"""SpecCLIP encoder tests against the real checkpoint. Skipped when the
checkpoint or the specclip package is unavailable."""
from pathlib import Path

import numpy as np
import pytest
import torch

from aspgapv3.models.encoders import SpecCLIPEncoder, default_specclip_ckpt

torch.manual_seed(0)


def ckpt_available():
    try:
        import specclip  # noqa: F401
    except ImportError:
        return False
    p = default_specclip_ckpt()
    return p is not None and p.is_file()


pytestmark = pytest.mark.skipif(not ckpt_available(), reason="SpecCLIP ckpt/pkg unavailable")


def test_frozen_load_and_encode():
    enc = SpecCLIPEncoder(unfreeze_last_k=0)
    assert all(not p.requires_grad for p in enc.parameters())
    x = torch.rand(4, 343) + 0.5
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (4, 768)
    assert torch.isfinite(z).all()
    # frozen -> deterministic
    with torch.no_grad():
        z2 = enc(x)
    assert torch.allclose(z, z2)


def test_partial_unfreeze_last2():
    enc = SpecCLIPEncoder(unfreeze_last_k=2)
    n_blocks = len(enc.backbone.blocks)
    for i, blk in enumerate(enc.backbone.blocks):
        want = i >= n_blocks - 2
        assert all(p.requires_grad == want for p in blk.parameters()), i
    assert all(p.requires_grad for p in enc.backbone.final_layernorm.parameters())
    # data embedding / positional embedding stay frozen
    assert not any(p.requires_grad for p in enc.backbone.data_embed.parameters())
    assert not any(p.requires_grad for p in enc.backbone.position_embed.parameters())
    n_train = len(enc.encoder_parameters() if hasattr(enc, "encoder_parameters") else [])
    assert n_train == 0  # wrapper itself has no extra params


def test_finetune_mode_gradient_flows():
    enc = SpecCLIPEncoder(unfreeze_last_k=2)
    x = torch.rand(2, 343) + 0.5
    z = enc(x)
    z.sum().backward()
    grads = [p.grad for p in enc.backbone.blocks[-1].parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    first = [p.grad for p in enc.backbone.blocks[0].parameters()]
    assert all(g is None for g in first)


def test_frozen_stays_eval_in_train_mode():
    enc = SpecCLIPEncoder(unfreeze_last_k=0)
    enc.train()
    assert not enc.backbone.training
    enc2 = SpecCLIPEncoder(unfreeze_last_k=2)
    enc2.train()
    assert enc2.backbone.training
    enc2.eval()
    assert not enc2.backbone.training


def test_mean_plus_stats_pooling():
    enc = SpecCLIPEncoder(unfreeze_last_k=0, pooling="mean_plus_stats")
    assert enc.out_dim == 1536
    x = torch.rand(3, 343) + 0.5
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (3, 1536)
    # stats part must differ from the mean part
    assert not torch.allclose(z[:, :768], z[:, 768:])
