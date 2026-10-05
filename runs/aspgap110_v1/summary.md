# evaluation — aspgap110_v1 (ckpt_best, epoch 19, val_loss -0.5458)


## iid_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 26201 223.2321 41.8815     138.5449 0.9115   135.7773             0.8880
   logg 26156   0.4225  0.0866       0.2921 0.8837     0.3411             0.8158
    m_h 26110   0.2139 -0.0424       0.1412 0.6422     0.1296             0.9886
alpha_m 26101   0.0689 -0.0135       0.0499 0.4542     0.0511             0.9552
   fe_h 25815   0.2164 -0.0555       0.1498 0.6025     0.1441             0.9814
   mg_h 25720   0.2126 -0.0603       0.1439 0.5107     0.1414             1.0089

## sky_test — per-label (raw units)

  label     n     rmse    bias  scatter_mad     r2  sigma_med  calibration_ratio
   teff 76342 225.0242 45.8925     136.6937 0.9085   130.8155             0.9433
   logg 76159   0.4047  0.1039       0.2675 0.8884     0.3139             0.8485
    m_h 76089   0.1989 -0.0341       0.1282 0.6853     0.1154             1.0237
alpha_m 76123   0.0640 -0.0103       0.0487 0.5124     0.0502             0.9097
   fe_h 74263   0.2015 -0.0477       0.1376 0.6531     0.1293             1.0072
   mg_h 73993   0.1992 -0.0509       0.1324 0.5859     0.1287             1.0281

## Conditional chemical information (beyond Teff/logg/[M/H] correlations)


**alpha_m** (conditional-mean R² on train = 0.549)

- residual recovery (all rows): R² = -0.1098, n = 101858

- residual recovery (label-valid rows): R² = -0.1098, pearson = 0.4633, n = 101858


**mg_fe** (conditional-mean R² on train = 0.480)

- residual recovery (all rows): R² = -0.1270, n = 99368

- residual recovery (label-valid rows): R² = -0.1270, pearson = 0.4197, n = 99368
