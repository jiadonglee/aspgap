# -*- coding: utf-8 -*-
"""
Author
------
JDL
Email
-----
jiadong.li at nyu.edu
Created on
----------
- Wed Feb 15 12:00:00 2023
Modifications
-------------
- Fri Feb 14 12:00:00 2023
Aims
----
- KVXP training script
"""

import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from kvxp.specformer import inferSpecFormer, SpecFormer2, SpecFormer
from kvxp.kvxp2 import CNN
from kvxp.data import  GXP_AP_4lb   

def cost_mse(pred, tgt):
    cost = torch.nn.MSELoss(reduction='mean')
    return cost(pred, tgt) 

def mask_ap(data, mask_band='ap'):
    if mask_band == 'ap':
        # data = torch.zeros_like(data, device=device)
        data = torch.ones_like(data, device=device)
    return data

def remove_prefix(state_dict, unwanted_prefix = '_orig_mod.'):
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    return state_dict


if __name__ == "__main__":

    #==============================
    #========== Hyper parameters
    """
    traing params
    """

    device = torch.device('cuda:0')
    BATCH_SIZE = int(2**8)
    num_epochs = 200
    part_train = False

    """
    data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap_xp_13286.npy"

    """
    model params
    """
    # INPUT_LEN = 110
    n_hi = 234
    n_cut = 7598
    n_enc = 110
    n_outputs = 4
    n_dim = 32
    # LR = 5e-5
    LR_ = 5e-4
    model_dir = "/data/jdli/gaia/model/0216_apxp/"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)

    mask_band = 'no'
    if mask_band == 'AP':
        mask_coef = torch.arange(n_enc, n_enc+n_hi, 1, dtype=int, device=device)
    else:
        mask_coef = None

    #=========================Data loading========================

    gdata  = GXP_AP_4lb(
        data_dir+tr_file,  
        part_train=part_train, device=device,
    )
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    #========================= Model ==============================

    def train_epoch(tr_loader, epoch):
            # model.train()
        model_infer.train()
        total_loss = 0.0
        start_time = time.time()
        itr=0
        
        for _, data in enumerate(tr_loader):
            x1 = data['x'][:,:n_enc]
            x2 = data['x'][:,n_enc:n_cut].reshape(-1, n_hi)
            # tgt = torch.zeros_like(data['y'])
            x2 = mask_ap(x2)

            _, attn = model_pre(x1, x2,mask_coef=None)
            # print(attn.shape)
            y = model_infer(attn)

            loss = cost_mse(y, data['y'].view(-1,4))
            loss_value = loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_infer.parameters(), 0.5)
            optimizer.step()

            total_loss+=loss_value
            itr+=1
            del data, y

        print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
        

    def eval(val_loader, epoch):
        model_infer.eval()
        total_val_loss=0
        itr=0

        # with torch.no_grad():
        for batch, data in enumerate(val_loader):
            x1 = data['x'][:,:n_enc]
            x2 = data['x'][:,n_enc:n_cut].reshape(-1, n_hi)
            x2 = mask_ap(x2)
            
            # tgt = torch.zeros_like(data['y'])
            _, attn = model_pre(x1, x2,mask_coef=None)
            y = model_infer(attn)

            loss = cost_mse(y, data['y'].view(-1, 4))
            total_val_loss+=loss.item()
            del data, y
            itr+=1

        print("val loss :%.4f"%(total_val_loss/itr))
        return total_val_loss/itr

#===================================================

    print("Training Start :================")

    for fold, (train_ids,valid_ids) in enumerate(kfold.split(gdata)):
        
        print(f'FOLD {fold}')
        print('--------------------------------')

        if fold==0:
            
            # model_infer = inferSpecFormer(n_enc+n_hi, n_outputs).to(device)

            model_infer = CNN(n_input=344, n_output=4, in_channel=32).to(device)
            model_pre = SpecFormer(n_enc, n_outputs, device=device).to(device)


            """
            =========================
            Loading pre-trained model
            =========================
            """         

            model_name = model_dir+"sp2_4l_%d_ep%d.pt"%(0,200)
            pre_dict = torch.load(model_name)
            pre_dict = remove_prefix(pre_dict)
            print(f"Loading pre-trained model: {model_name}")
            model_pre.load_state_dict(pre_dict)

            # model = torch.compile(model)
            optimizer = torch.optim.Adam(
                model_infer.parameters(), lr=LR_, weight_decay=1e-6
            )

            scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)


            for epoch in range(num_epochs+1):

                train_epoch(tr_loader, epoch)

                if epoch%5==0:
                    val_loss = eval(val_loader, epoch)
                    scheduler.step(val_loss)

                if epoch%50==0: 
                    save_point = "attn2_4l_%d_ep%d.pt"%(fold, epoch)
                    torch.save(model_infer.state_dict(), model_dir+save_point)

            # torch.cuda.empty_cache()
            del model_infer