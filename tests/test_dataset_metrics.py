import numpy as np
import pytest
import pandas as pd
import torch

from aspgapv3.data.dataset import APOGEEDataset, LabelScaler, load_catalog, save_catalog
from aspgapv3.metrics import metrics_table


def make_processed_root(tmp_path, n=120, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
    K = len(labels)
    y = rng.normal(0, 1, (n, K)).astype(np.float32)
    yerr = np.full((n, K), 0.05, dtype=np.float32)
    mask = rng.random((n, K)) > 0.25
    y[~mask] = np.nan
    yerr[~mask] = np.nan
    x110 = rng.normal(0.0, 1.0, (n, 110)).astype(np.float32)

    y_raw = y.copy()
    mean = np.nanmean(np.where(mask, y, np.nan), 0)
    std = np.nanstd(np.where(mask, y, np.nan), 0)
    scaler = LabelScaler.fit(y_raw, yerr, mask, np.full(K, 0.02), labels)
    scaler.save(tmp_path)
    np.savez(tmp_path / "labels.npz", y=y_raw, yerr=yerr, mask=mask,
             labels=np.array(labels, dtype="U16"))
    np.save(tmp_path / "coef110n.npy", x110)
    np.save(tmp_path / "coef110.npy", rng.normal(size=(n, 110)).astype(np.float32))

    splits = np.array(["train"] * 80 + ["val"] * 20 + ["iid_test"] * 20)
    rng.shuffle(splits)
    cat = pd.DataFrame({
        "row": np.arange(n),
        "source_id": np.arange(1000, 1000 + n),
        "snr": rng.uniform(50, 200, n),
        "split": splits,
        "teff": y_raw[:, 0], "logg": y_raw[:, 1], "m_h": y_raw[:, 2],
        "alpha_m": y_raw[:, 3], "fe_h": y_raw[:, 4], "mg_h": y_raw[:, 5],
        "is_mdwarf": rng.random(n) > 0.7,
    })
    save_catalog(cat, tmp_path)
    return tmp_path, labels, scaler


def test_scaler_roundtrip(tmp_path):
    root, labels, scaler = make_processed_root(tmp_path)
    s2 = LabelScaler.load(root)
    assert np.allclose(scaler.mean, s2.mean)
    assert scaler.labels == s2.labels == labels


def test_dataset_masks_and_normalization(tmp_path):
    root, labels, scaler = make_processed_root(tmp_path)
    ds = APOGEEDataset(root, split="train")
    assert len(ds) == (load_catalog(root)["split"] == "train").sum()
    item = ds[0]
    assert item["x"].shape == (110,)
    assert item["y"].shape == (6,) and item["mask"].dtype == torch.bool
    # normalized targets are 0 where masked
    assert (item["y"][~item["mask"]] == 0).all()
    # finite normalized values on the masked entries
    assert torch.isfinite(item["y"][item["mask"]]).all()
    with pytest.raises(FileNotFoundError):
        APOGEEDataset(root, split="train", input="does_not_exist")


def test_metrics_table():
    rng = np.random.default_rng(1)
    n = 100
    mu = rng.normal(0, 0.2, (n, 2))
    sig = np.full((n, 2), 0.1)
    y = rng.normal(0, 0.2, (n, 2))
    m = np.ones((n, 2), bool)
    m[:5, 0] = False
    t = metrics_table(mu, sig, y, m, ["a", "b"])
    assert t["n"].tolist() == [95, 100]
    assert np.isfinite(t["rmse"]).all()
