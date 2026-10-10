#!/usr/bin/env python
"""Pre-training sanity check of the red-clump A0 anchors (src/aspgapv3/rc_anchor.py), on the ext2 data root.

    python scripts/rc_anchor_check.py --out runs/aspgap_dist_ext2_rc/rc_anchor_check.md

Prints the selection / calibration, anchor counts per A_SFD bin, target-vs-SFD and target-vs-RJCE comparisons.
RJCE (Majewski+11): A_K = 0.918 (H - W2 - 0.08), A0 = A_K / 0.078 (K_s coefficient of the model), 2MASS/WISE quality-flagged,
rough check only (the intrinsic (H-W2)_0 of an RC star is ~0.05-0.1)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402
from aspgapv3.rc_anchor import A_SFD_FAC, RC_SET  # noqa: E402

BINS = [(-1, 0.5, "<=0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, 1e9, ">4")]


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def md(df):
    c = list(df.columns)
    rows = ["| " + " | ".join(map(str, c)) + " |", "|" + "|".join("---" for _ in c) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data/processed/aspcap_v0.6.0_ext2"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = {"enabled": True, "snr_min": 5.0, "sig_mu_floor": 0.1}
    tr = APOGEEDataset(a.root, "train", input="coef110n", labels=["teff", "logg", "m_h", "alpha_m"], astro=True,
                       require_xp_valid=True, rc_anchor=cfg)
    t = tr.astro_table
    ar = tr.astro_arr
    rc = tr.anchor_set == RC_SET
    asfd = A_SFD_FAC * t["ebv_sfd"].to_numpy(float)
    tgt, sig = ar["anc_t"].astype(float), ar["anc_s"].astype(float)
    # RJCE A0
    hw = t["h_m"].to_numpy(float) - t["w2mpro"].to_numpy(float)
    okr = t["tmass_ok_h"].to_numpy(bool) & t["wise_ok_w2"].to_numpy(bool) & np.isfinite(hw) & (t["w_qual"].astype(str).str[1:2] == "A").to_numpy() \
        if "w_qual" in t else np.isfinite(hw)
    a0_rjce = 0.918 * (hw - 0.08) / 0.078
    out = [f"# RC anchor check (train split, {a.root})\n", "## Calibration\n", "```json\n" + json.dumps(tr.rc_calib, indent=1) + "\n```\n",
           f"RC anchors: {int(rc.sum())} of {len(rc)} train stars; nearby {int((tr.anchor_set == 1).sum())}, "
           f"high-lat {int((tr.anchor_set == 2).sum())} (existing anchors unchanged; RC only where neither applies).\n"]
    rows = []
    for lo, hi, name in BINS:
        s = rc & (asfd > lo) & (asfd <= hi)
        r = {"A_SFD bin": name, "n_RC_anchor": int(s.sum()), "snr_5_10": int((s & (t["parallax"] / t["parallax_error"] < 10)).sum()),
             "median_sigma_A0": np.median(sig[s]) if s.any() else np.nan,
             "median_target": np.median(tgt[s]) if s.any() else np.nan, "median_A_SFD": np.median(asfd[s]) if s.any() else np.nan,
             "median(target-SFD)": np.median(tgt[s] - asfd[s]) if s.any() else np.nan,
             "MAD(target-SFD)": mad(tgt[s] - asfd[s])}
        s2 = s & okr
        d = tgt[s2] - a0_rjce[s2]
        r.update({"n_RJCE": int(s2.sum()), "median(target-RJCE)": np.median(d) if s2.any() else np.nan,
                  "median_RJCE": np.median(a0_rjce[s2]) if s2.any() else np.nan})
        rows.append(r)
    out += ["## Counts and target vs SFD / RJCE per A_SFD bin\n", md(pd.DataFrame(rows))]
    # high-latitude-like low extinction sanity: weighted mean of (target - SFD)
    lo_ = rc & (asfd < 0.1)
    w = 1 / sig[lo_] ** 2
    out.append(f"\nLow-extinction RC anchors (A_SFD<0.1, n={int(lo_.sum())}): weighted mean target - SFD = "
               f"{np.sum(w * (tgt[lo_] - asfd[lo_])) / w.sum():+.3f}, median {np.median(tgt[lo_] - asfd[lo_]):+.3f}, MAD {mad(tgt[lo_] - asfd[lo_]):.3f}\n")
    # target vs b (plane vs high latitude) for 0.5-2
    Path(a.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
