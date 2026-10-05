# aspgap_v3

**Gaia XP → APOGEE chemical label transfer for FGKM stars.**
SpecCLIP provides only the XP representation/pretraining; every supervised label
comes from APOGEE ASPCAP (Astra v0.6.0 `astraAllStarASPCAP`, HDU2). LAMOST never
provides labels here — it only participated in SpecCLIP's representation learning.

```
x_XP ──E_SpecCLIP──> z_768 ──MLP──> z_256 ──> H_atm  (raw teff, logg, [M/H])
                                    └──────> H_chem ([α/M], [Fe/H], [Mg/H], ...)
```

Each output is heteroscedastic `(mu, log sigma²)`; training uses a masked NLL
that folds in the APOGEE measurement error and a per-label sigma floor, so a
star only contributes to the labels it actually has (`M_ik` mask, never a
global `dropna`).

## The four benchmark models (Figure 2)

| config                  | encoder                                  | input          |
|-------------------------|------------------------------------------|----------------|
| `aspgap110.yaml`        | MLP (AspGap-style baseline)              | 110 XP coeffs  |
| `scratch.yaml`          | small transformer, random init           | 343-pixel XP   |
| `specclip_frozen.yaml`  | SpecCLIP XP encoder, frozen (Stage A)    | 343-pixel XP   |
| `specclip_ft.yaml`      | SpecCLIP, last-2 blocks unfrozen (B)     | 343-pixel XP   |

XP input convention (verified against SpecCLIP's own code and checkpoint):
sample the BP/RP system on `np.arange(336, 1021, 2)` nm (343 points), divide by
the flux at 550 nm (index 107), feed `[batch, 343, 1]`; the encoder prepends a
stats token → `[batch, 344, 768]`, mean-pooled over the token axis.

## Layout

```
configs/            base.yaml + one yaml per benchmark model
scripts/
  inspect_aspcap.py dump astraAllStarASPCAP HDU2 columns (first contact with the file)
  build_dataset.py  ASPCAP clean -> XP crossmatch -> 343-pixel spectra -> splits -> processed/
  train.py          train any config; masked NLL; binned sampler; Stage A/B/C via yaml
  evaluate.py       per-label/domain metrics, OOD scores, delta-alpha experiment
src/aspgapv3/
  labels/           schema.py (declarative label registry), loaders.py (DR21-ready)
  data/             xp.py, dataset.py, splits.py, sampler.py
  models/           encoders.py, heads.py, net.py
  losses.py metrics.py ood.py
tests/              synthetic-data tests, no APOGEE download needed
```

Processed dataset layout (`data/processed/<name>/`):

```
catalog.parquet|csv.gz  one row per Gaia source: labels, errors, masks, split,
                        sky_group, domain tags (metal_poor, mdwarf, fstar, faint_xp)
xp343.npy               float32 [n, 343]  SpecCLIP-grid, 550nm-normalized
coef110.npy             float32 [n, 110]  raw Gaia DR3 BP/RP coefficients
coef110n.npy            float32 [n, 110]  old-AspGap network input (G-normalized
                                          + RobustScaler): the baseline reads this
norm.npz                LabelScaler stats (train-set mean/std per label, sigma floors)
labels.npz              y / yerr / mask in raw units, label order
manifest.json           versions, cuts, counts, design-matrix provenance
```

Provenance of the XP coefficients: the local `gxp_chunk_*.hdf5` files store
the old AspGap *network input* (raw coefficients / 10^((15-G)*0.4), then
RobustScaler per photometer). `build_dataset.py` inverts both affine steps to
recover the raw DR3 coefficients (verified: corr(BP-RP, red/blue) = 0.96 on
bright stars), samples the 343-pixel spectra from those, and keeps the scaled
version verbatim as `coef110n.npy`.

## Environment

Everything runs in the existing `jepa311` conda env
(torch 2.5.1+cu124, specclip importable, checkpoint at
`../gaiaworld/specclip_assets/xp_encoder.ckpt`):

```bash
source ~/miniforge3/bin/activate jepa311   # adjust to this machine's path
pip install astropy pyarrow                # only needed once, for build scripts

# 0. look at the ASPCAP file before trusting any column mapping
python scripts/inspect_aspcap.py --fits <astraAllStarASPCAP-v0.6.0.fits> --hdu 2

# 1. build the SpecCLIP XP design matrix once (needs gaiaxpy, see data/xp.py)
python scripts/build_dataset.py --aspcap <fits> --xp-chunks ../aspgap/data/xpnorm \
    --out data/processed/aspcap_v0.6.0

# 2. Stage A on the frozen encoder
python scripts/train.py --config configs/specclip_frozen.yaml

# 3. Stage B (partial fine-tune) — see configs/specclip_ft.yaml
python scripts/train.py --config configs/specclip_ft.yaml
```

## Non-negotiables (mirrors the project definition)

1. Labels only from ASPCAP; per-label masks from per-element flags + valid
   ranges — never a `dropna` across all labels.
2. ASPCAP calibrated Teff embeds Gaia XP synthetic photometry (IRFM): train on
   `raw_teff` and never use calibrated Teff as an independent validation
   quantity.
3. One model for FGKM; M dwarfs are a performance regime, not another network.
4. XP only as input for the main model; parallax/photometry is a later ablation.
5. Predict the *published calibrated ASPCAP abundance scale*; no silent label
   "corrections".
6. `labels/loaders.py` is isolated so DR21 is a table swap, not a code change.
