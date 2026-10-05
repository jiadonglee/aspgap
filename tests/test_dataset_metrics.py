import numpy as np
import pytest
import pandas as pd
import torch

from aspgapv3.data.dataset import APOGEEDataset, LabelScaler, load_catalog, save_catalog
from aspgapv3.metrics import (derived_mgfe, fit_conditional_mean,
                             metrics_table, residual_recovery_score)


def make_processed_root(tmp_path, n=120, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
    K = len(labels)
    y = rng.normal(0, 1, (n, K)).astype(np.float32)
    yerr = np.full((n, K), 0.05, dtype=np.float32)
    mask = rng.random((n, K)) > 0.25
    y[~mask] = np.nan
    yerr[~mask] = np.nan
    x343 = rng.normal(1.0, 0.05, (n, 343)).astype(np.float32)

    y_raw = y.copy()
    mean = np.nanmean(np.where(mask, y, np.nan), 0)
    std = np.nanstd(np.where(mask, y, np.nan), 0)
    scaler = LabelScaler.fit(y_raw, yerr, mask, np.full(K, 0.02), labels)
    scaler.save(tmp_path)
    np.savez(tmp_path / "labels.npz", y=y_raw, yerr=yerr, mask=mask,
             labels=np.array(labels, dtype="U16"))
    np.save(tmp_path / "xp343.npy", x343)
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
    assert item["x"].shape == (343,)
    assert item["y"].shape == (6,) and item["mask"].dtype == torch.bool
    # normalized targets are 0 where masked
    assert (item["y"][~item["mask"]] == 0).all()
    # finite normalized values on the masked entries
    assert torch.isfinite(item["y"][item["mask"]]).all()
    with pytest.raises(FileNotFoundError):
        APOGEEDataset(root, split="train", input="does_not_exist")


def test_metrics_and_mgfe():
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

    mgfe, mgfe_e = derived_mgfe(mu[:, 0], sig[:, 0], mu[:, 1], sig[:, 1])
    assert np.allclose(mgfe, mu[:, 0] - mu[:, 1])
    assert np.allclose(mgfe_e, np.sqrt(2) * 0.1)


def test_conditional_residual_recovery():
    rng = np.random.default_rng(2)
    n = 2000
    teff = rng.normal(5000, 500, n)
    logg = rng.normal(2.5, 1.0, n)
    mh = rng.normal(-0.3, 0.4, n)
    # strong atmospheric correlation + a weak intrinsic component
    alpha = 0.1 * mh + 0.00005 * (teff - 5000) + 0.1 * rng.normal(size=n)
    f, delta, info = fit_conditional_mean(teff, logg, mh, alpha)
    assert info["r2_conditional_mean"] > 0.1
    # predicting delta with the true delta recovers R^2 = 1
    res = residual_recovery_score(delta, delta)
    assert res["r2_delta"] > 0.99
    # predicting delta with modest noise keeps most of the R^2
    res2 = residual_recovery_score(delta, delta + rng.normal(0, 0.05, n))
    assert 0.3 < res2["r2_delta"] < 1.0
    res3 = residual_recovery_score(delta, np.full(n, np.nan))
    assert res3["n"] == 0
