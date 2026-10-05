# evaluation — specclip_ft_v1 (ckpt_best, epoch 8, val_loss -0.3594)


## iid_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 26201 235.2417 52.9980     143.1477 0.9017   154.9445             0.6986
   logg 26156   0.4842  0.1592       0.3321 0.8472     0.4241             0.7798
    m_h 26110   0.2474 -0.0399       0.1824 0.5213     0.1897             0.6832
alpha_m 26101   0.0813 -0.0215       0.0615 0.2386     0.0705             0.8076
   fe_h 25815   0.2468 -0.0558       0.1914 0.4830     0.2076             0.7018
   mg_h 25720   0.2332 -0.0436       0.1757 0.4115     0.2199             0.7029

## sky_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 76342 215.4717 52.4763     134.0730 0.9161   143.4394             0.7125
   logg 76159   0.4290  0.1496       0.2915 0.8747     0.3791             0.8049
    m_h 76089   0.2151 -0.0206       0.1614 0.6319     0.1738             0.7460
alpha_m 76123   0.0709 -0.0142       0.0604 0.4018     0.0702             0.7282
   fe_h 74263   0.2143 -0.0333       0.1701 0.6078     0.1902             0.7287
   mg_h 73993   0.2027 -0.0195       0.1587 0.5709     0.2024             0.7039

## Conditional chemical information (beyond Teff/logg/[M/H] correlations)


**alpha_m** (conditional-mean R² on train = 0.549)

- residual recovery (all rows): R² = -0.4125, n = 101858

- residual recovery (label-valid rows): R² = -0.4125, pearson = 0.3382, n = 101858


**mg_fe** (conditional-mean R² on train = 0.480)

- residual recovery (all rows): R² = -0.7116, n = 99368

- residual recovery (label-valid rows): R² = -0.7116, pearson = 0.2478, n = 99368
