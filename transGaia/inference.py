from tqdm import tqdm
import torch
import numpy as np
import joblib

import sys
sys.path.append("/home/jdli/transpec")
from transGaia.transgaia import xp2label
from transGaia.data import GXP_5lb

scaler_labels = joblib.load('../docs/models/scaler_labels.gz')


def recover_label(y_hat, e_y_hat):
    y   = scaler_labels.inverse_transform((y_hat))
    e_y = scaler_labels.scale_*e_y_hat
    return y, e_y


def infer_4lbs_model_err(model_name, data_loader, 
                         n_input=113, n_output=8, n_dim=128, 
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
            # y_lst   = np.append(y_lst, data['y'].cpu().numpy())
            id_batch =  list(np.int64(data['id']))
            
            del output, data
            for idl in id_batch:
                id_lst.append(idl)

    out_lst, e_out_lst = np.array(out_lst).reshape(-1,4), np.array(e_out_lst).reshape(-1,4)
    y, e_y = recover_label(out_lst, e_out_lst)
    return {'labels':y, 'e_labels':e_y, 'source_id':id_lst}
