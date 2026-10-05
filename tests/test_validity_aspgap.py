"""require_xp_valid dataset filtering and the coef110n forward transform."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aspgapv3.data.dataset import APOGEEDataset
from aspgapv3.data.xp import aspgap_scaler_arrays, coef_to_aspgap_input

REAL = Path(__file__).resolve().parents[1] / "data/processed/aspcap_v0.6.0_test"


def make_root(tmp_path, n=40, n_bad=6):
    from aspgapv3.data.dataset import LabelScaler, save_catalog
    rng = np.random.default_rng(0)
    labels = ["teff", "logg"]
    cat = pd.DataFrame({"row": np.arange(n), "split": ["train"] * 20 + ["val"] * 10 + ["iid_test"] * 5 + ["sky_test"] * 5})
    save_catalog(cat, tmp_path)
    np.save(tmp_path / "xp343.npy", rng.normal(size=(n, 343)).astype(np.float32))
    np.save(tmp_path / "coef110n.npy", rng.normal(size=(n, 110)).astype(np.float32))
    bad = np.zeros(n, bool)
    bad[rng.choice(n, n_bad, replace=False)] = True
    np.save(tmp_path / "xpcov_valid.npy", ~bad)
    y, yerr, mask = rng.normal(size=(n, 2)), np.ones((n, 2)), np.ones((n, 2), bool)
    np.savez(tmp_path / "labels.npz", y=y, yerr=yerr, mask=mask, labels=np.array(labels))
    LabelScaler.fit(y, yerr, mask, np.array([0.1, 0.1]), labels).save(tmp_path)
    return bad, cat


@pytest.mark.parametrize("inp", ["xp343", "coef110n"])
def test_require_xp_valid_drops_bad_rows_in_all_splits(tmp_path, inp):
    bad, cat = make_root(tmp_path)
    tot = 0
    for split in ("train", "val", "iid_test", "sky_test"):
        d0 = APOGEEDataset(tmp_path, split, input=inp)
        d1 = APOGEEDataset(tmp_path, split, input=inp, require_xp_valid=True)
        rows = cat.row[cat.split == split].to_numpy()
        assert len(d0) == len(rows) and d0.n_dropped_invalid == 0
        assert d1.n_dropped_invalid == bad[rows].sum()
        assert len(d1) == len(rows) - bad[rows].sum()
        assert not bad[d1.indices].any() and len(d1.y) == len(d1)
        tot += d1.n_dropped_invalid
    assert tot == bad.sum()


def test_require_xp_valid_needs_file(tmp_path):
    make_root(tmp_path)
    (tmp_path / "xpcov_valid.npy").unlink()
    with pytest.raises(FileNotFoundError):
        APOGEEDataset(tmp_path, "train", require_xp_valid=True)


@pytest.mark.skipif(not (REAL / "coef110n.npy").is_file()
                    or not Path("../aspgap/model/scaler_bp_gmagand_0330.gz").is_file(),
                    reason="real test dataset / scalers unavailable")
def test_forward_transform_reproduces_coef110n():
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as td:
        sc = aspgap_scaler_arrays(td)
        assert (Path(td) / "aspgap_scaler.npz").is_file()
    raw = np.load(REAL / "coef110.npy")[:3000]
    ref = np.load(REAL / "coef110n.npy")[:3000]
    g = np.load(REAL / "side.npy")[:3000, 0] if (REAL / "side.npy").is_file() \
        else pd.read_parquet(REAL / "catalog.parquet")["phot_g_mean_mag"].to_numpy()[:3000]
    out = coef_to_aspgap_input(raw, g, sc)
    # raw was recovered from coef110n in float64 then stored float32: expect ~1e-5 rel
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
    import torch
    out_t = coef_to_aspgap_input(torch.as_tensor(raw), torch.as_tensor(g), sc)
    np.testing.assert_allclose(out_t.numpy(), out, rtol=1e-5, atol=1e-6)
