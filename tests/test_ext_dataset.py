"""External-label extension: hold-out design and label masking."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_ext_dataset as b  # noqa: E402

from aspgapv3.losses import MaskedHeteroscedasticNLL  # noqa: E402


def test_healpix_split_deterministic_and_by_pixel():
    rng = np.random.default_rng(0)
    pix = rng.integers(0, 12288, 50000)
    sid = (pix.astype(np.int64) << 49) + rng.integers(0, 1 << 49, 50000)
    assert (b.healpix32(sid) == pix).all()
    m1, m2 = b.healpix_train_mask(sid), b.healpix_train_mask(sid)
    assert (m1 == m2).all()
    assert abs(m1.mean() - 0.7) < 0.03
    for p in rng.choice(12288, 50):                 # every pixel is entirely train or entirely held out
        assert len(set(m1[pix == p])) <= 1


def test_cluster_split_stratified():
    rng = np.random.default_rng(1)
    feh = {f"c{i}": float(f) for i, f in enumerate(rng.uniform(-2.4, -0.1, 90))}
    feh["NGC 5139"] = -1.5
    held = b.split_clusters(feh)
    assert "NGC 5139" in held
    assert 0.28 < len(held) / len(feh) < 0.40
    # both sides cover the metal-poor end
    poor = sorted(feh, key=feh.get)[:12]
    assert any(c in held for c in poor) and any(c not in held for c in poor)
    assert held == b.split_clusters(feh)


def test_c_star_flags_nothing_for_normal_star():
    c, s = b.c_star(np.array([1.0]), np.array([1.2]), np.array([15.0]))
    assert abs(c[0]) < 3 * s[0] + 0.1


def test_missing_labels_are_masked_in_nll():
    """A GC star carries only [M/H]: the Teff/logg targets (garbage) and errors must not matter,
    and gradients w.r.t. the masked outputs must vanish."""
    n, k = 8, 3                                      # (teff, logg, m_h) as in the atm group
    mask = torch.zeros(n, k, dtype=torch.bool)
    mask[:, 2] = True
    y = torch.randn(n, k)
    yerr = torch.rand(n, k) * 0.1
    mu = torch.randn(n, k, requires_grad=True)
    lv = torch.zeros(n, k, requires_grad=True)
    loss = MaskedHeteroscedasticNLL(torch.ones(k), torch.full((k,), 0.05))
    l = loss(mu, lv, y, yerr, mask)
    l.backward()
    assert mu.grad[:, :2].abs().sum() == 0 and lv.grad[:, :2].abs().sum() == 0
    assert mu.grad[:, 2].abs().sum() > 0
    y2, e2 = y.clone(), yerr.clone()
    y2[:, :2], e2[:, :2] = 1e6, 1e6
    assert torch.allclose(l, loss(mu, lv, y2, e2, mask))
    # NaN targets after nan_to_num (as the dataset does) are equally harmless
    y3 = y.clone()
    y3[:, :2] = torch.nan_to_num(torch.full((n, 2), float("nan")))
    assert torch.allclose(l, loss(mu, lv, y3, yerr, mask))
    # a star with no labels at all in the chem group (alpha_m masked) contributes nothing
    lc = MaskedHeteroscedasticNLL(torch.ones(1), torch.full((1,), 0.05))
    z = torch.zeros(n, 1, dtype=torch.bool)
    assert float(lc(torch.randn(n, 1), torch.zeros(n, 1), torch.randn(n, 1), torch.ones(n, 1), z)) == 0.0


EXT = REPO / "data/processed/aspcap_v0.6.0_ext"
ROOT = REPO / "data/processed/aspcap_v0.6.0"


@pytest.mark.skipif(not (EXT / "catalog.parquet").exists(), reason="extended dataset not built")
def test_extended_root_integrity():
    c0 = pd.read_parquet(ROOT / "catalog.parquet", columns=["row", "source_id", "split"])
    c1 = pd.read_parquet(EXT / "catalog.parquet", columns=["row", "source_id", "split", "source"])
    n = len(c0)
    assert (c1.row.to_numpy() == np.arange(len(c1))).all()
    assert (c1.source_id.to_numpy()[:n] == c0.source_id.to_numpy()).all() and (c1.split.to_numpy()[:n] == c0.split.to_numpy()).all()
    e = c1.iloc[n:]
    assert (e.split == "train").all()
    assert not e.source_id.isin(c0.source_id).any()                     # none in any APOGEE split
    assert not e.source_id.duplicated().any()
    z = np.load(EXT / "labels.npz")
    assert len(z["y"]) == len(c1) and np.isfinite(z["y"][n:]).all() and np.isfinite(z["yerr"][n:]).all()
    gc = (e.source == "gc").to_numpy()
    mk = z["mask"][n:]
    assert mk[gc][:, 2].all() and not mk[gc][:, [0, 1, 3]].any()          # GC: [M/H] only
    vm = (e.source == "lamost_vmp").to_numpy()
    assert mk[vm][:, 2].all() and not mk[vm][:, [0, 1, 3]].any()
    sp = pd.read_parquet(REPO / "data/processed/ood/ood_split.parquet")
    for comp in ("galah", "cluster_gc", "lamost_vmp"):                    # a star is never train in its own component and held out in it
        s = sp[sp.component == comp]
        assert not set(s[s.role == "train"].source_id) & set(s[s.role == "heldout"].source_id)
