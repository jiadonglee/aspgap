import numpy as np
import pandas as pd
import pytest

from aspgapv3.data.splits import (assign_iid_splits, assign_sky_holdout,
                                  dedupe_max_snr, domain_tags)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "source_id": np.arange(n),
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-80, 80, n),
        "teff": rng.normal(5000, 500, n),
        "logg": rng.normal(3.0, 1.0, n),
        "m_h": rng.normal(-0.3, 0.4, n),
        "snr": rng.uniform(50, 200, n),
    })


def test_dedupe_keeps_max_snr():
    df = pd.DataFrame({
        "source_id": [1, 1, 2],
        "snr": [30.0, 90.0, 60.0],
        "val": ["a", "b", "c"],
    })
    d = dedupe_max_snr(df)
    assert len(d) == 2
    assert set(d["source_id"]) == {1, 2}
    assert d.loc[d["source_id"] == 1, "val"].iloc[0] == "b"


def test_iid_split_groups_never_overlap(toy):
    out = assign_iid_splits(toy, train=0.8, val=0.1, test=0.1, seed=1)
    ids = {s: set(out.loc[out["split"] == s, "source_id"]) for s in
           ("train", "val", "iid_test")}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["iid_test"])
    assert len(ids["train"]) + len(ids["val"]) + len(ids["iid_test"]) == len(toy)


def test_sky_holdout_removed_from_train(toy):
    out = assign_iid_splits(toy, seed=1)
    out = assign_sky_holdout(out, n_groups=16, holdout_groups=3, seed=2)
    sky = out.loc[out["split"] == "sky_test"]
    assert len(sky) > 0
    assert not (set(sky["source_id"]) &
                set(out.loc[out["split"] == "train", "source_id"]))
    # whole sky groups move together
    per_group = sky.groupby("sky_group").size()
    total_per_group = out.groupby("sky_group").size()
    assert (per_group == total_per_group[per_group.index]).all()


def test_domain_tags(toy):
    df = toy.copy()
    df["m_h"] = [-2.0, -0.2] * 200
    df["teff"] = [3500.0, 6500.0] * 200
    df["logg"] = [4.8, 2.0] * 200
    out = domain_tags(df)
    assert out["is_metal_poor"].sum() == 200
    assert out["is_mdwarf"].sum() == 200
    assert out["is_fstar"].sum() == 200
    assert not out["is_metal_poor_15"].any() or out["is_metal_poor_15"].sum() == 200
