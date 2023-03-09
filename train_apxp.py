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
- Fri Jan 31 12:00:00 2023
Modifications
-------------
- Fri Mar 6 12:00:00 2023
Aims
----
- training script
"""


import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from kvxp.data import GXP_AP_4lb
from kvxp.xpformer import XPformer2, MLP, MLP_upsampling
from kvxp.utils import *


##==================Model ======================================

def train_epoch(tr_loader, epoch, hallucinator, model_pre, decoder, encoder, opt_hal, opt_dec, opt_enc, alpha1=0.5, alpha2=0.5):
    hallucinator.train()
    decoder.train()
    encoder.train()
    total_loss = 0.0
    start_time = time.time()
    itr=0
    for batch, data in enumerate(tr_loader):
        y = data['y'].view(-1,4)
        xp = data['x'][:, :n_xp]
        # ap = data['x'][:, :n_xp]
        z  = encoder(xp)
        ap = hallucinator(z).reshape(-1,938,8)
        y1 = model_pre(ap)
        y2 = decoder(z)
        loss = alpha1*cost_mse(y1.view(-1,4), y) + alpha2*cost_mse(y2.view(-1,4), y)

        opt_hal.zero_grad()
        opt_dec.zero_grad()
        opt_enc.zero_grad()

        loss.backward()

        opt_hal.step()
        opt_dec.step()
        opt_enc.step()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(hallucinator.parameters(), 0.5)

        total_loss+=loss.item()
        itr+=1

    print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
    return total_loss/itr
    

def eval(val_loader, epoch, hallucinator, model_pre, decoder, encoder, 
         alpha1=0.5, alpha2=0.5):
    hallucinator.eval()
    model_pre.eval()
    decoder.eval()
    encoder.eval()

    total_val_loss_1=0
    total_val_loss_2=0

    itr=0
    for batch, data in enumerate(val_loader):
        y = data['y'].view(-1,4)
        xp = data['x'][:, :n_xp]
        # ap = data['x'][:, :n_xp]
        z  = encoder(xp)
        ap = hallucinator(z).reshape(-1,938,8)
        y1 = model_pre(ap)
        y2 = decoder(z)
        loss_1 = cost_mse(y1.view(-1,4), y)
        loss_2 = cost_mse(y2.view(-1,4), y)
    
        total_val_loss_1+=loss_1.item()
        total_val_loss_2+=loss_2.item()
        itr+=1

    print("val loss:%.4f"%(total_val_loss_1/itr))
    print("val loss:%.4f"%(total_val_loss_2/itr))
    return total_val_loss_1/itr



if __name__ == "__main__":

    #==============================
    #========== Hyper parameters===
    #==============================
    """
    traing params
    """
    band = "xp"
    mask_band = "ap"
    device = torch.device('cuda:0')
    BATCH_SIZE = int(2**8)
    num_epochs = 500
    part_train = False

    """
    data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "apspec_xp_173344.npy"
    """
    model params
    """

    # INPUT_LEN = 110
    n_xp = 110
    n_ap = 7504
    n_lat = 1024
    n_labels = 4
    # LR = 5e-5
    LR_ = 5e-4
    loss_penal = 2
    LMBDA_PEN = 1e-10
    LMBDA_ERR = 1e-1
    WEIGHT_DECAY = 1e-5
    alpha1 = 0.8
    alpha2 = 0.2
    # model_dir = f"/data/jdli/gaia/model/0220/"
    model_dir = f"/data/jdli/gaia/model/0309/"
    # save_preflix = f"xp2_4l_%d_ep%d.pt"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)
    else:
        print(f"save trained-model to {model_dir}")

    #=========================Data loading ================================
    gdata  = GXP_AP_4lb(
        data_dir+tr_file,
        part_train=part_train, 
        device=device,
    )
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    #======================================================================
    tr_loss_lst = []
    val_loss_lst = []

    print("Training Start :================")     

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(gdata)):
        
        print(f'FOLD {fold}')
        print('--------------------------------')

        if fold==0:

            # model_pre = MLP(n_ap, n_labels, hidden_size=1024).to(device)
            model_pre = XPformer2(938, n_labels, device=device, input_proj=False).to(device)
            model_pre_name = "/data/jdli/gaia/model/0306/" +"ap2_4l_%d_ep%d.pt"%(0,1000)

            model_pre.load_state_dict(
                remove_prefix(
                    torch.load(model_pre_name)
                )
            )
            print(f"loading pre-trained model {model_pre_name}")

            # Freeze the weights of the pre-trained model
            for param in model_pre.parameters():
                param.requires_grad = False

            hallucinator = MLP_upsampling(n_lat, n_ap, dropout=0.5).to(device)
            decoder = MLP(n_lat, n_labels).to(device)
            encoder = MLP_upsampling(n_xp, n_lat, dropout=0.5).to(device)

            print()

            opt_hal = torch.optim.Adam(hallucinator.parameters(), lr=LR_, weight_decay=WEIGHT_DECAY)
            opt_dec = torch.optim.Adam(decoder.parameters(), lr=LR_, weight_decay=WEIGHT_DECAY)
            opt_enc = torch.optim.Adam(encoder.parameters(), lr=LR_, weight_decay=WEIGHT_DECAY)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader  = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

            for epoch in range(num_epochs+1):
                tr_loss = train_epoch(tr_loader, epoch, 
                            hallucinator=hallucinator, model_pre=model_pre, decoder=decoder, encoder=encoder,
                            opt_hal=opt_hal, opt_dec=opt_dec, opt_enc=opt_enc, 
                            alpha1=alpha1, alpha2=alpha2)
                tr_loss_lst.append(tr_loss)

                if epoch%5==0:
                    val_loss = eval(val_loader, epoch,  
                                    hallucinator=hallucinator, model_pre=model_pre, decoder=decoder, encoder=encoder)
                    val_loss_lst.append(val_loss)

                if epoch%50==0: 
                    # save_point = save_preflix%(fold, epoch)
                    torch.save(decoder.state_dict(), model_dir+f"lat2_4l_%d_ep%d.pt"%(fold, epoch))
                    np.save("check/loss.npy", {'tr_loss':tr_loss_lst, 'val_loss':val_loss_lst})
