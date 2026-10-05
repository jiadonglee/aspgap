# PROGRESS

## 2026-10-05 (later) — bad-spectra exclusion, coef110n degradation, queue v3

- `data.require_xp_valid: true` (base.yaml): APOGEEDataset drops rows with
  `xpcov_valid == False` (non-positive 550 nm flux; 1,628 of 601,291 in the
  full set) from every split and input type, so all runs share one star set;
  per-split drop counts are in the `data` log event. `false` = old behaviour.
- Degraded eval now also covers the `coef110n` baseline: degraded raw
  coefficients are pushed through the exact forward AspGap transform
  (divide by 10^((15-G')*0.4), RobustScaler center/scale cached in
  `aspgap_scaler.npz`); a test reproduces coef110n.npy from coef110.npy.
- `.gitignore`: `/data/` so `src/aspgapv3/data/` is tracked.
- Queue `scripts/run_benchmarks_v3.sh` (_r2 runs, log logs/benchmarks_v3.log):
  pixel_fusion, aspgap110, specclip_ft, pixel_fusion_noaug, specclip_frozen,
  pixel_fusion_phot, scratch; each followed by evaluate.py with degraded mode.

## 2026-10-05 — pixel-space fusion, XP noise products, noise-matched augmentation (step 1)

Goal: prepare for Gaia DR4 with a pixel-space model that keeps AspGap's two
real advantages (absolute flux, XP noise covariance) without DR3 coefficient
scaling, and train on bright stars degraded to faint-star noise.

Built (see README layout): `scripts/build_xp_noise.py`, `data/xpnoise.py`,
`augment.py` (noise-match), `models/side.py` (FiLM on trunk features, zero-init
output), `AttnPool` in `SpecCLIPEncoder` (`pooling: attn`), `amp.py`,
`evaluate.py --degrade-g`, `configs/pixel_fusion{,_noaug}.yaml`, tests.

AMP NaN fix: all transformer runs had `train_loss=nan` under fp16 autocast.
Now bf16 autocast (`train.amp_dtype`), mu/logvar cast to fp32 and both NLLs
evaluated outside autocast, non-finite loss / grad-norm batches are skipped and
counted (`n_nonfinite` per epoch in log.jsonl). Eval uses the same policy.
Smoke run (pixel_fusion, 2 epochs, 30k test set, batch 128): n_nonfinite = 0.

Noise data (30k test set, 28,517 stars; raw h5 routed by `source_id >> 43`,
which is what the file-name index range encodes):
- raw h5 coefficients vs `coef110.npy` (inverted from AspGap chunks): max
  element-wise relative difference 7.6e-3 (on a near-zero coefficient; p99.99
  3.8e-5); max |dc| / max_k|c_k| per star = 7.4e-8, median 4e-12.
- per-pixel sigma vs gaiaxpy 2.1.4 `calibrate()` flux_error (20 stars, env
  `xpbuild`): max relative difference 7.0e-8, median 2.1e-8; flux agrees to
  5.9e-8 of star max.
- Cholesky: 0 of 28,517 stars needed jitter (BP and RP), 0 failures, 0 stars
  missing in the raw files.
- Convention found while verifying: gaiaxpy multiplies sqrt(D^T cov_gx D) by
  `standard_deviation`, so the physical coefficient covariance is
  std^2 * cov_gx = diag(e) corr diag(e). `../aspgap` XPSampler.sample omits this
  std^2 (and has w_rp[:,None]*w_rp[:,None] in the RP pixel covariance); neither
  is reproduced here.
- Bank: 77 bins x 200 donors (G 10.0-17.7) from a 1% random subsample of the
  219.1M-star XP catalog; G histogram of the full catalog stored.

Smoke result (2 epochs only, not a science result): see runs/pixel_fusion_smoke
(summary.md, metrics_degraded_*.csv; the run dir is not committed): [Fe/H] RMSE iid_test 0.25 native ->
0.28 / 0.32 / 0.41 / 0.47 at G' = 15 / 16 / 17 / 17.5.

Known approximations: donor SED != star SED; donor noise not rescaled within
its 0.1 mag bin; xp343_sigma treats the 550 nm normalizer as noise-free; bp_rp
unchanged by augmentation; stars with non-positive post-noise f550 are skipped.
Note: `.gitignore` has `data/`, which also hides `src/aspgapv3/data/` (never
tracked); the edits to `data/xp.py`, `data/dataset.py` and new `data/xpnoise.py`
do not appear in `git status`.

## 2026-10-04 — full dataset built; benchmark queue launched

Full build: 1,095,480 ASPCAP rows -> 744,371 clean spectra -> 671,817 unique
sources -> **601,291 stars with recovered XP** (89.5% crossmatch; the missing
10% are sources absent from the old gxp_chunk coverage).

```
train 472,395 | val 26,353 | iid_test 26,201 | sky_test 76,342
label coverage: alpha_m 99.6%, fe_h 98.3%, mg_h 98.0%
metal-poor 18,902 | [M/H]<-1.5: 6,897 | M dwarfs 16,072 | F stars 47,816
G = 10.2 / 13.0 / 16.0 (p5/50/95)
```

Physics checks on xp343: log(red/blue) vs Teff monotonic across
3000-8000 K (0.95 -> -0.53 in 500K bins); FGK-only spearman -0.79;
0.1% noise rows (non-positive blue mean) — negligible, kept flagged
implicitly by their spectra.

Benchmark queue running on the A100 (`scripts/run_benchmarks_v1.sh`, ~6 min
per epoch at batch 256, binned sampler, AMP):

1. `specclip_frozen_v1` — epoch 0: val RMSE teff 278 K, logg 0.60,
   [M/H] 0.326, **[α/M] 0.094**, [Fe/H] 0.295, [Mg/H] 0.287 ( dex).
2. `specclip_ft_v1` (last-2 blocks, lr_enc 1e-5) queued
3. `scratch_v1` queued
4. `aspgap110_v1` queued

Monitor: `grep event=epoch /tmp/opencode/benchmarks_v1.log | tail -5`
(run dirs: runs/<name>/, final metrics in runs/<name>/metrics_*.csv +
summary.md via scripts/evaluate.py after each finishes).

## 2026-10-04 — real data: astraAllStarASPCAP v0.6.0 downloaded, schema aligned, dataset build validated

- `astraAllStarASPCAP-0.6.0.fits.gz` (11.7 GB) downloaded from DR20 SAS
  (`spectro/astra/0.6.0/summary/`), sha1 verified against the official
  sha1sum, decompressed to 2.27 GB (memmap-able, per-column lazy reads).
- `scripts/inspect_aspcap.py` on the real file (1,095,480 rows): the schema's
  runtime column resolution now matches it everywhere. Key real names:
  `m_h_atm`/`alpha_m_atm` (atmospheric fit), `e_*` errors (not `*_err`),
  `<el>_h` + `e_<el>_h` + `<el>_h_flags` per element, `raw_*` = FERRE,
  `teff` = **IRFM-calibrated** (V band from Gaia XP synthetic photometry —
  the leak the project definition warned about), `irfm_teff` = the target.
- Flag policy fixed to the official bitmask: bits 5-8 (censored x3 + bad grid
  edge) always mask; warn bits 9-11 mask when `labels.mask_warn_bits: true`;
  upper-limit bits 0-4 do NOT mask (my initial guess of bits 0-3 was wrong).
- gaiaxpy 2.1.4 in an isolated `xpbuild` conda env built the (55,343) BP/RP
  design matrices once (3.4 s, cached to npz; jepa311 keeps numpy 1.26.4).
  Its design matrix carries physical-flux normalization (~1e-19), so the
  550nm-degeneracy check must be scale-free.
- The old `gxp_chunk_*.hdf5` store the AspGap *network input* (raw DR3
  coefficients / 10^((15-G)*0.4), RobustScaler-transformed). Both steps are
  affine — `build_dataset.py` inverts them exactly to recover raw
  coefficients (verified: corr(BP-RP, red/blue) = 0.96, smooth spectra),
  samples xp343 from raw, and keeps the scaled input as `coef110n.npy` for
  the baseline. 3,385 chunks; 95% of ASPCAP stars crossmatched.
- Test build (30k-star slice) validated end to end: label coverage 99-100%
  per label (per-label masking works; a global dropna would be far worse),
  monotonic log(red/blue) vs Teff across 3000-8000 K, teff_calibrated vs
  raw_teff corr 0.996 (median offset 3 K), splits + domain tags written.

Full 30k-test counts: 1,095,480 rows -> 744,371 clean spectra -> 671,817
unique sources -> 28,517/30,000 XP-matched in the slice.

Next: full 671k build (running), then Stage A on frozen SpecCLIP.

## 2026-10-04 — repo scaffolded

**Smoke-tested end to end on this machine (synthetic data, real SpecCLIP
checkpoint):** `specclip_frozen` (Stage A), `specclip_ft` (Stage B, 14.2M
encoder params trained at 1e-5 while heads run at 3e-4), and `aspgap110`
(coef110 input + binned sampler) all complete train → checkpoint → iid_test +
sky_test metrics → `evaluate.py` (per-label/domain tables, OOD scores,
Δ[α/M] + Δ[Mg/Fe] conditional-recovery experiment). 34/34 tests green
(`jepa311`: `python -m pytest tests -q`).

Verified on this machine before writing any code:

- SpecCLIP repo + `xp_encoder.ckpt` (490 MB) live at
  `../gaiaworld/specclip_assets/`; `jepa311` env has torch 2.5.1+cu124 (1 GPU)
  and imports specclip (editable install of that repo).
- Checkpoint loads and encodes `(4, 343) -> (4, 768)`; 42.8M params. Encoder is
  `SpecFormerControl1_stats`: per-spectrum normalization + prepended stats
  token -> `[b, 344, 768]`; mean-pool over tokens is SpecCLIP's downstream
  convention (`gaiaworld/gaiaworld/src/gaiaworld_simulator/xp/specclip_adapter.py`
  already encodes exactly this convention; we reimplement it as a trainable
  wrapper so Stage B/C can unfreeze blocks).
- XP input convention cross-checked against SpecCLIP's own
  `downstream-with-demo/predict_xp_wclip_v0.py`: grid `np.arange(336, 1021, 2)`
  nm (343 px), normalize by flux at index 107 (= 550 nm).
- Old AspGap (`../aspgap`): works on 110 XP coefficients; its
  `data/xpnorm/gxp_chunk_*.hdf5` files hold `source_id` + 110 coefficients +
  metadata for the Gaia DR3 XP sample — reusable for the ASPCAP crossmatch
  without touching the Gaia archive. Its `src/xp_utils.py` shows the exact
  design-matrix convention (`coef @ dm`, dm from gaiaxpy
  `__generate_xp_matrices_and_merge`, BP/RP merged with per-pixel weights).

Built:

- Declarative label registry (`labels/schema.py`): Tier-1 = teff(raw), logg,
  [M/H], [α/M], [Fe/H], [Mg/H]; per-label sigma floors + valid ranges; runtime
  column resolution so astra v0.6.0 -> DR21 is a data swap.
- Masked heteroscedastic NLL with APOGEE errors + sigma floors; per-label
  masks; binned weighted sampler `p ∝ 1/sqrt(N_bin)` over (teff, logg, [M/H]);
  source_id-grouped IID split + spherical-kmeans sky holdout + domain tags
  (metal-poor, M dwarf, F star, faint-XP).
- Three encoders behind one interface (all output 768-d): frozen/partially
  fine-tuned SpecCLIP, scratch transformer on 343 px, AspGap-style MLP on 110
  coefficients.
- `scripts/build_dataset.py / train.py / evaluate.py / inspect_aspcap.py`,
  configs for the four benchmark models, synthetic-data tests.

Not yet done (deliberately): download astraAllStarASPCAP v0.6.0, finalize the
column mapping via `inspect_aspcap.py` (the label registry resolves columns
defensively, but first contact with the real table must confirm), install
gaiaxpy once for the design-matrix step, then the first real Stage A run. No
science result yet — everything above is scaffolding + tests.

Bugs found and fixed during the smoke runs (all now covered by tests):
- per-head NLL: atm and chem heads need separate floor/weight buffers;
- `fit_conditional_mean`: `finite` lived on the ok-subset and leaked into a
  full-length `np.where` (shape 725 vs 535);
- `predict` closure inside `fit_conditional_mean` allocated `len(D_)` instead
  of the full input length;
- config key was `ckpt`, encoder kwarg is `ckpt_path` — unified.

## Gate

- tests: `python -m pytest tests -q` (jepa311) — must stay green (34 passing).
