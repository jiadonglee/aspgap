# evaluation — scratch_v1 (ckpt_best, epoch 1, val_loss 0.1117)


## iid_test — per-label (raw units)

  label     n     rmse     bias  scatter_mad      r2  sigma_med  calibration_ratio
   teff 26201 389.2673 100.4640     295.0854  0.7309   307.1192             0.9155
   logg 26156   0.6340   0.0258       0.4804  0.7381     0.6072             0.8629
    m_h 26110   0.4222  -0.1500       0.3390 -0.3938     0.3440             0.9066
alpha_m 26101   0.1042  -0.0452       0.0681 -0.2499     0.0952             1.0094
   fe_h 25815   0.3855  -0.1330       0.3286 -0.2615     0.3285             0.9342
   mg_h 25720   0.3781  -0.1390       0.3222 -0.5476     0.3421             0.8656

## sky_test — per-label (raw units)

  label     n     rmse     bias  scatter_mad      r2  sigma_med  calibration_ratio
   teff 76342 350.7024 101.9439     269.6487  0.7779   264.3514             0.9422
   logg 76159   0.5986   0.0546       0.4345  0.7560     0.5448             0.9239
    m_h 76089   0.3534  -0.0823       0.3024  0.0062     0.3063             0.9187
alpha_m 76123   0.0944  -0.0380       0.0647 -0.0615     0.0928             0.9217
   fe_h 74263   0.3407  -0.0779       0.3037  0.0081     0.2961             0.9414
   mg_h 73993   0.3391  -0.0832       0.2954 -0.2000     0.3054             0.8991

## Conditional chemical information (beyond Teff/logg/[M/H] correlations)


**alpha_m** (conditional-mean R² on train = 0.549)

- residual recovery (all rows): R² = -1.4508, n = 101858

- residual recovery (label-valid rows): R² = -1.4508, pearson = 0.0955, n = 101858


**mg_fe** (conditional-mean R² on train = 0.480)

- residual recovery (all rows): R² = -1.5429, n = 99368

- residual recovery (label-valid rows): R² = -1.5429, pearson = 0.0568, n = 99368
