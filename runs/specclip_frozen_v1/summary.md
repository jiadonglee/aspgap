# evaluation — specclip_frozen_v1 (ckpt_best, epoch 19, val_loss -0.2238)


## iid_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 26201 274.5069 26.4705     151.3496 0.8662   141.6809             0.8580
   logg 26156   0.5243  0.1852       0.3563 0.8209     0.4695             0.7876
    m_h 26110   0.2766 -0.0946       0.1840 0.4018     0.2106             0.6750
alpha_m 26101   0.0813 -0.0104       0.0621 0.2385     0.0701             0.7683
   fe_h 25815   0.2739 -0.1042       0.1908 0.3633     0.2204             0.6986
   mg_h 25720   0.2636 -0.0999       0.1881 0.2478     0.2363             0.6018

## sky_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 76342 241.2813 33.8581     140.4384 0.8949   126.4013             0.8678
   logg 76159   0.4817  0.1812       0.3140 0.8420     0.4167             0.8201
    m_h 76089   0.2300 -0.0638       0.1597 0.5793     0.1883             0.6856
alpha_m 76123   0.0722 -0.0038       0.0599 0.3784     0.0701             0.7083
   fe_h 74263   0.2294 -0.0695       0.1649 0.5504     0.1956             0.7212
   mg_h 73993   0.2236 -0.0677       0.1687 0.4782     0.2140             0.6402

## Conditional chemical information (beyond Teff/logg/[M/H] correlations)


**alpha_m** (conditional-mean R² on train = 0.549)

- residual recovery (all rows): R² = -0.4498, n = 101858

- residual recovery (label-valid rows): R² = -0.4498, pearson = 0.3054, n = 101858


**mg_fe** (conditional-mean R² on train = 0.480)

- residual recovery (all rows): R² = -0.6341, n = 99368

- residual recovery (label-valid rows): R² = -0.6341, pearson = 0.2139, n = 99368
