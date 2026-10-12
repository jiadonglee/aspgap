# aspgap_v3

Gaia XP BP/RP coefficients -> **APOGEE-scale Teff, logg, [M/H], [alpha/M]** plus **M_G** (intrinsic absolute G
magnitude) and **A0** (extinction at 550 nm), hence a distance, for FGKM stars down to G = 17.65.
A successor of AspGap v1: same 110 G-normalized coefficients as input, but one network for stellar labels and
distances, trained with noise-matched augmentation and extra training data beyond APOGEE.
Labels come from APOGEE ASPCAP (Astra v0.6.0 `astraAllStarASPCAP`, HDU2; Teff is the raw FERRE value, since the
published IRFM-calibrated Teff embeds Gaia XP photometry).

## Model (`configs/aspgap_dist_ext2.yaml`, run `aspgap_dist_ext_r2`)

```
110 coefs (G-normalized, RobustScaled) -> MLP (110-512-768) -> trunk (768-256-256) -+-> H_atm  (Teff, logg, [M/H])
                                                                                     +-> H_chem ([alpha/M])
                                                                                     +-> H_dist (M_G, sigma_MG, A0 >= 0)
```

* Every label is heteroscedastic `(mu, log sigma^2)`; masked NLL with the APOGEE error and a per-label sigma floor,
  so a star only contributes to labels it has (GC / VMP stars carry only [M/H]). 738k parameters.
* **Distance**: no brightness enters the network. The parallax likelihood (Gaia DR3, Lindegren+21 zero-point, all stars,
  no S/N cut) ties `M_G + 0.789 A0` to the observed G and parallax; a 2MASS/WISE photometric forward model
  (intrinsic-colour net of the four labels only + fixed extinction coefficients) constrains A0; A0 anchors (nearby
  stars A0 = 0, high-latitude stars A0 = SFD) break the colour-label degeneracy. Distance modulus at inference:
  `mu = G - M_G - 0.789 A0`. See `src/aspgapv3/models/distance.py`.
* **Noise-matched augmentation** (`augment.py`): bright APOGEE stars are degraded to the XP noise of fainter stars
  (G' drawn from the Gaia XP magnitude histogram, donor covariances from a bank of random XP stars), p = 0.5.
  Checkpoints are selected on the mean of clean and degraded validation loss. Uniform (random) sampler.

## Training data

APOGEE (S/N > 50, Teff < 8000 K, one row per source; 470,941 train rows) plus 141,384 train-only external rows
(`scripts/build_ext_dataset.py`, root `data/processed/aspcap_v0.6.0_ext2`):
GALAH DR4 stars (Teff, logg, [M/H], [alpha/M] on the APOGEE scale), globular-cluster members ([M/H] only, cluster-level
value; E(B-V) <= 0.5, PM / parallax consistent, <= 100 per cluster) and LAMOST very-metal-poor stars ([M/H] only).

Hold-out design: APOGEE `iid_test` (5%, random) and `sky_test` (32 spherical-kmeans sky groups held out whole), plus a
validation split; selection and early stopping use APOGEE validation only. For the OOD sets, GALAH and LAMOST VMP are
split by HEALPix (nside 32, 70/30) and globular clusters by whole cluster (sorted by Harris [Fe/H], one of every three
held out, omega Cen included); open clusters, LAMOST hot stars and StarHorse are never trained on.
`eval_ood.py --holdout` scores only the never-trained rows (`ood_split.parquet`).

## How to run

Environment `jepa311` (torch, pandas, pyarrow, astropy, h5py, joblib; build steps also need `gaiaxpy`, the Gaia
`zero_point` package, `dustmaps` with the SFD maps in `data/dustmaps/sfd`, and network access to the Gaia TAP).
On shared nodes use at most half of `nproc` (dataloader workers included) and set `OMP_NUM_THREADS` etc.

```bash
# data (each step writes next to the previous one)
python scripts/build_dataset.py --aspcap <astraAllStarASPCAP-v0.6.0.fits> --xp-chunks ../aspgap/data/xpnorm \
       --out data/processed/aspcap_v0.6.0                      # labels, splits, coef110n / coef110
python scripts/build_xp_noise.py --data data/processed/aspcap_v0.6.0 --nproc 8   # covariances, noise bank
python scripts/build_astrometry_phot.py --root data/processed/aspcap_v0.6.0      # Gaia / 2MASS / WISE
python scripts/build_ood.py                                    # OOD sets: GALAH, OC, GC, LAMOST hot / VMP, StarHorse
python scripts/build_ext_dataset.py --nproc 8 --threads 4      # APOGEE + external rows -> ..._ext2

# train + evaluate (CONFIG selects the model; configs/aspgap_dist.yaml = APOGEE only)
CONFIG=configs/aspgap_dist_ext2.yaml scripts/run_aspgap_dist.sh aspgap_dist_ext_r2
python scripts/train.py --config configs/aspgap_dist_ext2.yaml --set run.name=NAME run.out_dir=runs   # train only
python scripts/evaluate.py --run runs/NAME [--out-dir DIR]     # per-label / domain / distance tables, degraded-G' curve
python scripts/eval_ood.py --run NAME --holdout                # OOD metrics -> runs/NAME/ood_holdout
python scripts/ext_summary.py --runs aspgap_dist_r3 aspgap_dist_ext_r2 --out aspgap_dist_ext_r2   # comparison
python -m pytest tests                                         # synthetic-data tests
```

Layout: `configs/` (base, aspgap_dist, aspgap_dist_ext2), `scripts/` (build_*, train, evaluate, eval_ood, ext_summary),
`src/aspgapv3/` (`models/` encoder, heads, distance; `data/` dataset, splits, XP / noise utilities; `labels/`
ASPCAP registry and loaders; `augment.py`, `losses.py`, `metrics.py`, `dist_eval.py`, `ood.py`), `tests/`.
`runs/` keeps only small text outputs (checkpoints and `data/` are git-ignored).

## Results (run `aspgap_dist_ext_r2`, single seed)

APOGEE test sets, RMSE vs ASPCAP (same stars; AspGap v1 = `runs/aspgap_v1_ref`):

| split | model | Teff [K] | logg | [M/H] | [alpha/M] |
|---|---|---|---|---|---|
| iid (26,112) | AspGap v1 | 162.0 | 0.242 | 0.131 | 0.0623 |
| iid | **aspgap_v3 ext_r2** | **105.3** | **0.194** | **0.105** | **0.0422** |
| sky (76,333) | AspGap v1 | 164.0 | 0.228 | 0.117 | 0.0584 |
| sky | **aspgap_v3 ext_r2** | **98.1** | **0.176** | **0.097** | **0.0393** |

Adding GALAH / GC / VMP data changes the APOGEE numbers by < 1.5% relative to the APOGEE-only run (r3).

Distance (iid test, parallax S/N > 10): M_G residual MAD 0.21 mag (RMSE 0.38), parallax pull robust-std 0.85. Median
predicted fractional distance error: 10% (G < 11), 15% (G 13-14), 22% (14-15), 30% (15-16), 41% (16-17); 14% overall.

OOD, held-out rows only (model - external label; r3 = APOGEE-only model in brackets):

| set | n | result |
|---|---|---|
| GALAH | 54,146 | RMSE Teff 83 K (91), logg 0.128 (0.142), [M/H] 0.086 (0.089), [alpha/M] 0.040 (0.043) |
| globular clusters (42 held out) | 23,439 | [M/H] bias -0.13 (+0.29), RMSE 0.42 (0.51) |
| LAMOST VMP | 1,520 | [M/H] bias +0.11 (+0.60), RMSE 0.44 (0.76) |
| open clusters (50) | 9,481 | [M/H] bias -0.25 (-0.20), RMSE 0.45 (0.38); distance +0.08 mag at G 15-16, +0.37 at 16-17, +1.0 at G > 17 |
| LAMOST hot (Teff > 7000 K) | 91,282 | Teff bias -1650 K (saturation), [M/H] bias -0.10 (+0.06) |

Known limitations:
* Teff saturates near 8000 K (the APOGEE cut): hotter stars are predicted far too cool.
* Distances of faint (G > 16) and highly extinguished stars are too large (open clusters: median +0.4 mag at G 16-17,
  +1.0 mag at G > 17; globular clusters +0.6 mag overall).
* A0 is underestimated at high extinction (open clusters, A_SFD > 2: A0 - A_V(cat) = -0.63 mag median).
* The predicted [alpha/M] sigma is not a good OOD / error flag (AUC for |err| > 3 MAD: 0.71 on GALAH); [M/H] sigma is
  uninformative for GC (0.52) and VMP (0.57) stars.
* Faint open-cluster [M/H] carries an extra bias of about -0.05 relative to the APOGEE-only model (-0.24 vs -0.20 at G 16-17).
* Rejected experiment: red-clump A0 anchors (`aspgap_dist_ext2_rc`, code in git history) moved A0 closer to references
  but did not improve distances and worsened APOGEE label RMSE by 5-8%.

All earlier experiments (SpecCLIP encoders, pixel-space fusion, auxiliary element labels, v1-latent distillation,
binned sampler, ...) are preserved in the git history before the slimming commit.
