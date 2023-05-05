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
- specformer model script
"""


import time
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
# from kvxp.xpformer import XPformer2
from kvxp.apxp import CNN
from kvxp.data import XPAP4l
from kvxp.utils import *

##==================Model ======================================

def train_epoch(tr_loader, epoch, model, opt1, data_type='ap'):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    itr=0
    for batch, data in enumerate(tr_loader):

        if data_type == 'ap':
            x = data['ap'].reshape(-1, 1, n_ap*n_dim)

        elif data_type == 'xp':
            x = data['xp']

        y = model(x)
        loss1 = cost_mse(y.view(-1,4), data['y'].view(-1,4))
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        total_loss+=loss1.item()
        itr+=1

    print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
    

def eval(val_loader, epoch, model, data_type='ap'):
    model.eval()
    total_val_loss=0
    itr=0
    # with torch.no_grad():
    for batch, data in enumerate(val_loader):

        if data_type == 'ap':
            x = data['ap'].reshape(-1, 1, n_ap*n_dim)

        elif data_type == 'xp':
            x = data['xp']

        y = model(x)
        loss1 = cost_mse(y.view(-1,4), data['y'].view(-1,4))
        loss_value = loss1.item()
        total_val_loss+=loss_value
        itr+=1

    print("val loss:%.4f"%(total_val_loss/itr))
    return total_val_loss/itr


if __name__ == "__main__":

    #==============================
    #========== Hyper parameters
    """
    traing params
    """

    device = torch.device('cuda:0')
    TOTAL_NUM = 200
    BATCH_SIZE = int(2**10)
    num_epochs = 500
    part_train = False

    """
    data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "apspec_xp_173344.dump"
    """
    model params
    """

    n_enc = 11
    n_dim = 64
    n_xp  = 110
    n_ap  = 128
    n_labels = 4
    # n_cut = n_hi*n_dim + n_enc
    n_outputs = 4
    n_head =  8
    n_layer = 8
    LR_ = 1e-4
    # LMBDA_ERR = 1e-1
    model_dir = "/data/jdli/gaia/model/0311_mlp/"
    pre_trained = False
    # loss_function = WeightedMSE(10.0)
    loss_function = cost_mse

    save_preflix = f"ap2_4l_%d_ep%d.pt"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)
    else:
        print(f"save trained-model to {model_dir}")

    #=========================Data loading ================================

    gdata  = XPAP4l(data_dir+tr_file, device=device, part_train=False)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    #======================================================================


    print("Training Start :================")     

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(gdata)):
        
        print(f'FOLD {fold}')
        print('--------------------------------')

        if fold==0:

            model = CNN(n_input=n_ap*n_dim, n_output=n_labels).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=LR_, weight_decay=1e-6
            )

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

            for epoch in range(num_epochs+1):
                train_epoch(tr_loader, epoch, model, optimizer)

                if epoch%5==0:
                    val_loss = eval(val_loader, epoch, model)

                if epoch%50==0: 
                    save_point = save_preflix%(fold, epoch)
                    torch.save(model.state_dict(), model_dir+save_preflix%(fold, epoch))