import sys
sys.path.append("/home/jdli/transpec")
import torch
import numpy as np
from transGaia.transgaia import xp2label
from tqdm import tqdm


def retrieve_id(data_loader):
    id_lst = []
    for data in tqdm(data_loader):
        id_batch =  list(np.int64(data['id']))
        for idl in id_batch:
            id_lst.append(idl)
    return id_lst


def predict_label_attn(data_loader, model, n_label=1, attn=True):
    out_lst = np.array([])
    # attn_all = np.array([])
    attn_all, id_lst = [], []

    with torch.no_grad():

        for data in tqdm(data_loader):
            
            id_batch =  list(np.int64(data['id']))
            for idl in id_batch:
                id_lst.append(idl)

            if attn:
                output, att = model(data['x'])
                out_lst = np.append(out_lst, output.cpu().numpy())
                attn_all.append([a.cpu().numpy().reshape(-1,113,113) for a in att])
                # attn_batch = np.array([a.cpu().numpy().reshape(-1,113,113) for a in att])
                # attn_all = np.append(attn_all, attn_batch.reshape(-1,8,113,113))
            else:
                output = model(data['x'])
                out_lst = np.append(out_lst, output.cpu().numpy())
                attn_all = None
                
            del output, data
                
    out_lst = np.array(out_lst).reshape(-1, n_label)

    return {'labels':out_lst, 'source_id':id_lst, 'attn':attn_all}


def recover_labels(labels_post):
    norm = np.array([1e-2, 15., 10., 50.]).astype(np.float32)
    shift_norm = np.array([25., 25., 25., 25.]).astype(np.float32)
    return (labels_post-shift_norm)/norm

def recover_errors(errors):
    norm = np.array([1e-2, 15., 10., 50.]).astype(np.float32)
    return errors/norm



def infer_4lbs_model(model_name, data, n_labels=4, device='cpu'):

    model = xp2label(
            n_encoder_inputs=113,
            n_outputs=4, channels=128, n_heads=8, n_layers=8, attn=False,
        ).to(device)
    model.load_state_dict(torch.load(model_name))

    out_lst = np.array([])
    y_lst = np.array([])
    id_lst  = []
    
    if type(data) == torch.utils.data.dataloader.DataLoader:
    
        with torch.no_grad():
            for data in tqdm(data):
                output = model(data['x'])
                out_lst = np.append(out_lst, output.cpu().numpy())
                y_lst = np.append(y_lst, data['y'].cpu().numpy())
                
                id_batch =  list(np.int64(data['id']))
                
                for idl in id_batch:
                    id_lst.append(idl)
            del output, data

        torch.cuda.empty_cache()
        lbs_raw   = np.array(out_lst).reshape(-1, n_labels)
        trues_raw = np.array(y_lst).reshape(-1, 5)
        preds = recover_labels(lbs_raw)

    elif type(data) == np.ndarray:
        with torch.no_grad():
            lbs_raw = model(torch.tensor(data.astype(np.float32)).to(device))
        torch.cuda.empty_cache()

        preds = recover_labels(lbs_raw.cpu().numpy())
        trues_raw = []
        
    return {'labels':preds, 'true':trues_raw, 'source_id':id_lst}

    
def infer_4lbs_model_err(model_name, data, n_labels=4, device='cpu'):
    
    model = xp2label(
            n_encoder_inputs=113,
            n_outputs=2*n_labels, channels=128, n_heads=8, n_layers=8, attn=False,
        ).to(device)
    model.load_state_dict(torch.load(model_name))

    out_lst = np.array([])
    e_out_lst = np.array([])
    y_lst = np.array([])
    id_lst  = []
    
    if type(data) == torch.utils.data.dataloader.DataLoader:
    
        with torch.no_grad():
            for data in tqdm(data):
                output = model(data['x'])
                out_lst = np.append(out_lst, output[:,:4].cpu().numpy())
                e_out_lst = np.append(e_out_lst, output[:,4:].cpu().numpy())

                y_lst = np.append(y_lst, data['y'].cpu().numpy())
                id_batch =  list(np.int64(data['id']))
                
                for idl in id_batch:
                    id_lst.append(idl)
            del output, data

        torch.cuda.empty_cache()
        lbs_raw   = np.array(out_lst).reshape(-1, n_labels)
        e_lbs_raw = np.array(e_out_lst).reshape(-1, n_labels)
        trues_raw = np.array(y_lst).reshape(-1, 5)
        preds = recover_labels(lbs_raw)
        e_preds = recover_errors(e_lbs_raw)

    elif type(data) == np.ndarray:
        with torch.no_grad():
            output = model(torch.tensor(data.astype(np.float32)).to(device))

        torch.cuda.empty_cache()

        preds = recover_labels(output[:,:4].cpu().numpy())
        e_preds = recover_errors(output[:,4:].cpu().numpy())
        trues_raw = []
        
    return {'labels':preds, 'e_labels':e_preds, 'true':trues_raw, 'source_id':id_lst}    
    
    