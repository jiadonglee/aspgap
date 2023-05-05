from tqdm import tqdm
import torch
import numpy as np
import joblib

import sys
sys.path.append("/home/jdli/transpec")
from kvxp.transgaia import xp2label
from kvxp.data import GXP_5lb
from astroquery.gaia import Gaia
from tqdm import tqdm
import pandas as pd
import time



def chunks(lst, n):
    ""
    "Split an input list into multiple chunks of size =< n"
    ""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
        


def download_gaiacoeffs(source_id):
    dl_threshold = 5000 # DataLink server threshold. It is not possible to download products for more than 5000 sources in one single call.
    ids          = source_id
    ids_chunks   = list(chunks(ids, dl_threshold))
    datalink_all = []

    print(f'* Input list contains {len(ids)} source_IDs')
    print(f'* This list is split into {len(ids_chunks)} chunks of <= {dl_threshold} elements each')
    
    retrieval_type = 'XP_CONTINUOUS'
    data_structure = 'COMBINED'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3' # Options are: 'Gaia DR3' (default), 'Gaia DR2'
    dl_key         = f'{retrieval_type}_{data_structure}.xml'

    datalink_all = []

    # dl_keys  = [inp for inp in datalink.keys()]
    # dl_keys.sort()

    ii = 0
    for chunk in ids_chunks:
        ii+=1
        start = time.time()
        print(f'Downloading Chunk #{ii}; N_files = {len(chunk)}')
        datalink  = Gaia.load_data(
            ids=chunk, data_release = data_release,
            data_structure=data_structure
            )
        datalink_all.append(datalink)
        print(f"Downloading Chunk time #{(time.time() - start)/60} mins")
        
    
    product_list_tb  = [item for sublist in datalink_all for item in sublist[dl_key]]
    product_list_ids = [sid for sublist in datalink_all for item in sublist[dl_key] for sid in item.array["source_id"].data]
    
    N_COEFF = 55
    bp_coef, rp_coef = [], []
    e_bp_coef, e_rp_coef = [], []
    
    for kk,tab in tqdm(enumerate(product_list_tb)):
    
        bp_coefficients         = np.array([d.data for d in tab.array["bp_coefficients"]]).reshape(-1, N_COEFF)
        bp_coefficients_errors  = np.array([d.data for d in tab.array["bp_coefficient_errors"]]).reshape(-1, N_COEFF)

        rp_coefficients         = np.array([d.data for d in tab.array["rp_coefficients"]]).reshape(-1, N_COEFF)
        rp_coefficients_errors  = np.array([d.data for d in tab.array["rp_coefficient_errors"]]).reshape(-1, N_COEFF)

        bp_coef.append(bp_coefficients)
        rp_coef.append(rp_coefficients)

        e_bp_coef.append(bp_coefficients_errors)
        e_rp_coef.append(rp_coefficients_errors)

    bp_coef = np.vstack(tuple([_ for _ in bp_coef]))
    rp_coef = np.vstack(tuple([_ for _ in rp_coef]))
    e_bp_coef = np.vstack(tuple([_ for _ in e_bp_coef]))
    e_rp_coef = np.vstack(tuple([_ for _ in e_rp_coef]))

    print(bp_coef.shape, rp_coef.shape, e_bp_coef.shape, e_rp_coef.shape)
    print(len(product_list_ids))
    
    df_xpcontinous = pd.DataFrame(
    {'source_id':product_list_ids, 'bp_coef':bp_coef.tolist(), 
     'e_bp_coef':e_bp_coef.tolist(), 'rp_coef':rp_coef.tolist(),  
     'e_rp_coef':e_rp_coef.tolist()}
    )
    return df_xpcontinous
     

    
def norm_xpcontinous(bp_raw, rp_raw, gmag, source_id, 
                     bp_model_name='../docs/models/scaler_bp_gmagand_robust.gz',
                     rp_model_name='../docs/models/scaler_rp_gmagand_robust.gz'):

    mag_norm = 10**((15.-gmag)*0.4)
    
    norm_bp_mag = bp_raw/mag_norm[:,None]
    norm_rp_mag = rp_raw/mag_norm[:,None]
    
    scaler_bp = joblib.load(bp_model_name)
    scaler_rp = joblib.load(rp_model_name)
    
    norm_bp = scaler_bp.transform(norm_bp_mag)
    norm_rp = scaler_rp.transform(norm_rp_mag)
    
    return pd.DataFrame({'source_id':source_id, 'bp':norm_bp.tolist(), 'rp':norm_rp.tolist()})


def norm_tm_photo(J, H, K, J_norm=13.859):   
    # J_norm = 13.118
    # J_norm = 13.859
    norm_photo = np.c_[J,H,K]/J_norm
    return norm_photo

    
def recover_label(y_hat, e_y_hat, model_name='../docs/models/scaler_labels.gz'):
    
    scaler_labels = joblib.load(model_name)
    y   = scaler_labels.inverse_transform((y_hat))
    e_y = scaler_labels.scale_*e_y_hat
    return y, e_y

def recover_scale_label(y_hat, e_y_hat, bias=2, scale_ext=10):
    scaler_labels = joblib.load('../docs/models/scaler_labels.gz')
    
    y   = scaler_labels.inverse_transform(y_hat/scale_ext-bias)
    e_y = (e_y_hat * scaler_labels.scale_)/scale_ext
    return y, e_y

def recover_dist_label(y_hat, e_y_hat):
    norm = np.array([1e-2, 15., 10., 50.])
    shift_norm = np.array([25., 25., 25., 25.])
    y = (y_hat-shift_norm)/norm
    e_y = e_y_hat/norm
    return y, e_y


def infer_4lbs_model_err(model_name, data_loader,transcale_method=recover_scale_label, n_input=113, n_output=8, n_dim=128,
n_head=8, n_layer=8, device=torch.device('cuda:0')):
    
    model = xp2label(n_encoder_inputs=n_input, n_outputs=n_output, channels=n_dim, n_heads=n_head, n_layers=n_layer).to(device)
    model.load_state_dict(torch.load(model_name))
    
    out_lst, e_out_lst = np.array([]), np.array([])
    id_lst = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            output = model(data['x'])
            out_lst   = np.append(out_lst,   output[:,:4].cpu().numpy())
            e_out_lst = np.append(e_out_lst, output[:,4:].cpu().numpy())
            id_batch =  list(np.int64(data['id']))
            
            del output, data
            for idl in id_batch:
                id_lst.append(idl)

    out_lst, e_out_lst = np.array(out_lst).reshape(-1,4), np.array(e_out_lst).reshape(-1,4)

    y, e_y = transcale_method(out_lst, e_out_lst)
    return {'labels':y, 'e_labels':e_y, 'source_id':id_lst}
