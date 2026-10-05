#!/usr/bin/env python
"""Build auxiliary element-abundance labels for an already-processed dataset.

    python scripts/build_aux_labels.py --root data/processed/aspcap_v0.6.0 \
        [--aspcap data/raw/astraAllStarASPCAP-0.6.0.fits] [--elements c_h n_h ...]

Writes NEW files only (refuses to overwrite unless --force); labels.npz /
norm.npz / catalog are never touched:
    labels_aux.npz   y / yerr / mask [n, A] raw units (NaN where masked), labels,
                     source_id [n] (row-aligned with catalog `row`)
    norm_aux.npz     LabelScaler stats (TRAIN split only), same format as norm.npz
    aux_manifest.json  per-element coverage (all stars / train split), floors

Cleaning = exactly what build_dataset.py did (same global cuts from
build_config.yaml, same max-S/N dedupe, same flag policy and per-label masks),
then rows are matched to the processed catalog by gaia source_id. Alignment is
verified by rebuilding the six main labels and comparing with labels.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.config import load_config  # noqa: E402
from aspgapv3.data.dataset import LabelScaler, load_catalog  # noqa: E402
from aspgapv3.labels.loaders import dedupe_max_snr, load_and_clean  # noqa: E402
from aspgapv3.labels.schema import AUX_CANDIDATES, spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aspcap", default=None, help="default: manifest.json['aspcap']")
    ap.add_argument("--elements", nargs="*", default=list(AUX_CANDIDATES))
    ap.add_argument("--hdu", type=int, default=None, help="default: manifest.json['hdu'] or 2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_files = [root / f for f in ("labels_aux.npz", "norm_aux.npz", "aux_manifest.json")]
    if not args.force and any(f.exists() for f in out_files):
        raise SystemExit(f"{[str(f) for f in out_files if f.exists()]} exist; use --force")
    man = json.loads((root / "manifest.json").read_text())
    cfg = load_config(root / "build_config.yaml")
    aspcap = args.aspcap or man["aspcap"]
    main_labels = list(man["labels"])
    aux = [a for a in args.elements if a not in main_labels]
    teff_variant = cfg["data"].get("teff_variant", cfg["labels"].get("teff_variant", "raw"))
    print(f"[1/3] loading {aspcap}: main={main_labels} aux={aux}", flush=True)
    res = load_and_clean(aspcap, main_labels + aux,
                         snr_min=float(cfg["data"]["snr_min"]),
                         teff_max=float(cfg["data"]["teff_max"]),
                         teff_variant=teff_variant,
                         mask_warn_bits=bool(cfg["labels"].get("mask_warn_bits", True)),
                         hdu=args.hdu if args.hdu is not None else int(man.get("hdu", 2)))
    df = res["df"]
    keep = dedupe_max_snr(df, res["resolved"]["source_id"])
    sid_all = df[res["resolved"]["source_id"]].to_numpy()[keep].astype(np.int64)
    y, yerr, mask = res["y"][keep], res["yerr"][keep], res["mask"][keep]
    nm = len(main_labels)

    # ---- align to the processed catalog by source_id ----------------------
    cat = load_catalog(root).sort_values("row").reset_index(drop=True)
    assert np.array_equal(cat["row"].to_numpy(), np.arange(len(cat)))
    sid_cat = cat["source_id"].to_numpy().astype(np.int64)
    pos = pd.Index(sid_all).get_indexer(sid_cat)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} catalog stars not found in the cleaned table")
    y, yerr, mask = y[pos], yerr[pos], mask[pos]

    # ---- verify alignment with the main labels ----------------------------
    z = np.load(root / "labels.npz")
    assert [str(s) for s in z["labels"]] == main_labels
    same_mask = np.array_equal(mask[:, :nm], z["mask"])
    same_y = np.allclose(np.nan_to_num(y[:, :nm]), np.nan_to_num(z["y"]), atol=1e-4)
    print(f"[2/3] alignment check vs labels.npz: masks equal={same_mask}, values equal={same_y}",
          flush=True)
    if not (same_mask and same_y):
        raise SystemExit("aux labels do NOT reproduce labels.npz for the main labels; aborting")

    ya, ea, ma = y[:, nm:], yerr[:, nm:], mask[:, nm:]
    train = (cat["split"] == "train").to_numpy()
    floors = np.array([spec(a).sigma_floor for a in aux])
    scaler = LabelScaler.fit(ya[train], ea[train], ma[train], floors, aux)
    np.savez(out_files[0], y=ya, yerr=ea, mask=ma, labels=np.array(aux, dtype="U16"),
             source_id=sid_cat)
    scaler.save(root, "norm_aux.npz")
    cov = {a: {"n_all": int(ma[:, k].sum()), "frac_all": float(ma[:, k].mean()),
               "n_train": int(ma[train, k].sum()), "sigma_floor": float(floors[k]),
               "median_err": float(np.nanmedian(ea[ma[:, k], k])) if ma[:, k].any() else None}
           for k, a in enumerate(aux)}
    out_files[2].write_text(json.dumps({"aspcap": aspcap, "n_stars": int(len(cat)),
                                        "coverage": cov}, indent=2))
    print("[3/3] coverage (frac of all stars with a valid label):", flush=True)
    for a, c in sorted(cov.items(), key=lambda kv: -kv[1]["frac_all"]):
        print(f"  {a:8s} {100 * c['frac_all']:6.1f}%  n_train={c['n_train']}", flush=True)


if __name__ == "__main__":
    main()
