import pandas as pd
import numpy as np
import time 
from tqdm import tqdm
from astroquery.gaia import Gaia



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
        datalink  = Gaia.load_data(ids=chunk,data_release = data_release, 
                                   retrieval_type=retrieval_type, format='votable',
                                   data_structure=data_structure)
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


def l2norm(y):
    return np.sqrt(np.sum(y**2, axis=1))