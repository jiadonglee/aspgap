import pandas as pd
import numpy as np
from astropy.table import Table, join
from astropy.io import fits
from tqdm import tqdm
from sklearn import preprocessing
import joblib

def recover_coef(series):
    return [list(map(float, s[1:-1].split(','))) for s in series]


def l2norm(y):
    return np.sqrt(np.sum(y**2, axis=1))

bp_scaler_name = "/home/jdli/transpec/models/scaler_bp_gmagand_0228.gz"
rp_scaler_name = "/home/jdli/transpec/models/scaler_rp_gmagand_0228.gz"
label_scaler_name = "/home/jdli/transpec/models/scaler_labels_0228.gz"

coef_names = ['bp_coefficients', 'bp_coefficient_errors', 
              'rp_coefficients', 'rp_coefficient_errors']

pred_dfs = pd.DataFrame()
df = Table.read("data/GaiaDR3_SID_G16.fits").to_pandas()
xp_coeff_array = np.zeros([4, len(df), 55])

xp_chunks = pd.read_csv(
    "/nfsdata/share/gaiaxp/gdr3_jdli_sid_xp_continuous_mean_spectrum.csv", 
    chunksize=1000000, sep="|")


save_dir = "/nfsdata/share/gaiaxp/"
save_name = "xp_2tian.npy"


for i,xp_chunk in tqdm(enumerate(xp_chunks)):
    
    pred_df  = pd.merge(df, xp_chunk, left_on='source_id', right_on='source_id')
    pred_dfs = pd.concat((pred_dfs, pred_df), axis='index')
    print(pred_dfs.shape)

for col in coef_names:
    pred_dfs[col] = recover_coef(pred_dfs[col])
    
for i in range(4):
    xp_coeff_array[i] = np.array(
        [np.array(x) for x in pred_dfs[coef_names[i]].values]
    )

    
bp_snr = np.abs(xp_coeff_array[0,:,:]/xp_coeff_array[1,:,:])
rp_snr = np.abs(xp_coeff_array[2,:,:]/xp_coeff_array[3,:,:])

mask_bp = bp_snr>1
mask_rp = rp_snr>1
gmag_norm = 10**((15.-pred_dfs['phot_g_mean_mag'].values)*0.4)


norm_bp = xp_coeff_array[0,:,:]/gmag_norm[:,None]
norm_rp = xp_coeff_array[2,:,:]/gmag_norm[:,None]

scaler_bp = joblib.load(bp_scaler_name)
scaler_rp = joblib.load(rp_scaler_name)


snr_bp_global = l2norm(xp_coeff_array[0,:,:])/l2norm(xp_coeff_array[1,:,:])
snr_rp_global = l2norm(xp_coeff_array[2,:,:])/l2norm(xp_coeff_array[3,:,:])


save_data = {
    "x":np.c_[norm_bp, norm_rp],
    "x_mask":np.c_[bp_snr<1, rp_snr<1],
    "source_id":pred_dfs['source_id'].values
}
np.save(save_dir+save_name, save_data)
