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
from kvxp.kvxp2 import specformer2, inference
from kvxp.data import  GXP_AP_4lb


def chi(pred, tgt, e_pred, e_tgt):
    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    return (pred-tgt)/var

def kl_divergence(mean_pred, std_pred, mean2=0, std2=1):
    var_pred = torch.square(std_pred)
    # var2 =  torch.square(std2)
    var2 = std2**2
    return 0.5*(torch.log(var2/var_pred) + (var_pred + (mean_pred - mean2)**2)/var2 - 1.)


def cost_penalty(pred, tgt, e_pred, e_tgt, model, lmbda=1e-10, lmbda_2=1):
    """
    Computes the loss between the predictions and targets, with a penalty term.
    """
    err_min, err_max = 1e-4, 20

    chi_pred = chi(pred, tgt, e_pred, e_tgt)
    kl_pen = lmbda_2*kl_divergence(chi_pred.mean(axis=0), chi_pred.std(axis=0)).sum()

    cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
    # jitter = f_jitter * tgt
    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    loss = cost(pred, tgt, var)
    # Add the penalty term
    # penalty = lmbda * sum(abs(p).sum() for p in model.parameters())

    # print(penalty.cpu().detach().numpy())
    loss +=  kl_pen
    return loss

def cost_val(pred, tgt, e_pred, e_tgt):
    err_min, err_max = 1e-4, 200

    var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
    cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
    loss = cost(pred, tgt, var)
    return loss

def cost_mse(pred, tgt):
    cost = torch.nn.MSELoss(reduction='mean')
    return cost(pred, tgt)


if __name__ == "__main__":

    #==============================
    #========== Hyper parameters
    """
    traing params
    """

    device = torch.device('cuda:0')
    TOTAL_NUM = 1000
    BATCH_SIZE = int(2**9)
    num_epochs = 200
    part_train = False

    """
    data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap_xp_13286.npy"

    # coef_mask = None # no mask in training
    coef_mask = torch.arange(108, 386, 1, dtype=int,  device=device)
    # coef_mask = torch.arange(0,   110, 1, device=device) # XP mask in training
    # coef_mask = torch.arange(110, 220, 1, device=device) # AP mask in training
    """
    model params
    """

    # INPUT_LEN = 110
    n_hi = 7514
    n_start = 0
    n_end   = 110
    n_enc = 110
    n_outputs = 4
    n_dim = 32
    n_head =  4
    n_layer = 2
    # LR = 5e-5
    LR_ = 5e-4
    LMBDA_PEN = 1e-10
    LMBDA_ERR = 1e-1
    model_dir = "/data/jdli/gaia/model/0214_apxp/"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)

    #=========================Data loading========================

    gdata  = GXP_AP_4lb(
        data_dir+tr_file, total_num=TOTAL_NUM, 
        part_train=part_train, device=device,
    )
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    #========================= Model ==============================

    def train_epoch(tr_loader, epoch):
            # model.train()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        itr=0
        
        for batch, data in enumerate(tr_loader):
            x = data['x']

            output = model(x).view(-1, 4)
            loss = cost_mse(output, data['y'].view(-1,4))
            loss_value = loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss+=loss_value
            itr+=1
            del data, output

        print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
        

    def eval(val_loader, epoch, mask_band='ap'):
        model.eval()
        total_val_loss=0
        itr=0

        with torch.no_grad():
            for batch, data in enumerate(val_loader):

                x = data['x']
                if mask_band == 'ap':
                    x[:, 110:] = 0.
                # output = inference(model, x, torch.arange(108, 386, 1, device=device), device)
                output = model(x, inf=True).view(-1, 4)
                loss = cost_mse(output, data['y'].view(-1, 4))
                total_val_loss+=loss.item()
                del data, output
                itr+=1

        print("val loss:%.4f"%(total_val_loss/itr))
        return total_val_loss/itr

#===========================================================

    print("Training Start :================")

    for fold, (train_ids,valid_ids) in enumerate(kfold.split(gdata)):
        
        print(f'FOLD {fold}')
        print('--------------------------------')

        if fold==0:

            model = specformer2(n_enc, n_outputs).to(device)
            # model = torch.compile(model)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=LR_, weight_decay=1e-5
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
                    save_point = "sp2_4l_%d_ep%d.pt"%(fold, epoch)
                    torch.save(model.state_dict(), model_dir+save_point)

            # torch.cuda.empty_cache()
            del model