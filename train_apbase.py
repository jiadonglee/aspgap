# -*- coding: utf-8 -*-
"""
Author
------
JDL
Email
-----
jdli at nao.cas.cn
Created on
----------
- Fri Jan 31 12:00:00 2023
Modifications
-------------
- Fri Mar 6 12:00:00 2023
Aims
----
- tuning script
"""

import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from kvxp.data import GXP_4lb, XPAP4l
from kvxp.xpformer import CNN, MLP
from kvxp.utils import *


##==================Model ======================================

def train_epoch(tr_loader, epoch, model, opt):

    model.train()
    total_loss = 0.0
    start_time = time.time()
    itr=0

    for batch, data in enumerate(tr_loader):

        y = data['y'].view(-1,4)
        # xp = data['x']

        y2 = model(data['ap'])

        # loss = cost_penalty(y2[:,:4], y, y2[:,4:], data['e_y'])
        loss = cost_mse(y2[:,:4], y) 

        opt.zero_grad()
        loss.backward()
        opt.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        total_loss+=loss.item()
        itr+=1

    print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
    return total_loss/itr
    

def eval(val_loader, epoch, model):
    model.eval()

    total_val_loss_2=0

    itr=0
    for batch, data in enumerate(val_loader):

        y = data['y'].view(-1,4)
        # xp = data['x']
        # y2 = model(xp)
        y2 = model(data['ap'])

        loss_2 = cost_mse(y2[:,:4], y)    
        total_val_loss_2+=loss_2.item()
        itr+=1

    print("val loss:%.4f"%(total_val_loss_2/itr))
    return total_val_loss_2/itr



if __name__ == "__main__":

    #==============================
    #========== Hyper parameters===
    #==============================
    """
    traing params
    """
    device = torch.device('cuda:1')
    BATCH_SIZE = int(2**11)
    num_epochs = 500
    part_train = True

    """
    data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "apspec_xp_cut_0415.dump"
    """
    model params
    """

    # INPUT_LEN = 110
    n_xp = 110
    # n_ap = 128
    n_ap = 8575
    n_dim = 64
    n_lat = 1024
    n_labels = 4
    # LR = 5e-5
    LR_ = 1e-5
    loss_penal = 2
    LMBDA_PEN = 1e-10
    LMBDA_ERR = 1e-1
    WEIGHT_DECAY = 1e-5
    PRE_TRAINED = False
    alpha1 = 0.8
    alpha2 = 0.2
    model_dir = "/data/jdli/gaia/model/0416/"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print(f"make dir {model_dir}")
        os.makedirs(model_dir)
    else:
        print(f"save trained-model to {model_dir}")

    #=========================Data loading ================================

    gdata  = XPAP4l(data_dir+tr_file, device=device, part_train=False)
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

            model = MLP(n_ap, n_labels).to(device)
            # model = CNN(n_ap, n_labels).to(device)

            if PRE_TRAINED:

                model.load_state_dict(
                    remove_prefix(
                        torch.load(model_dir+f"ap2_4l_%d_ep%d.pt"%(fold, 50))
                    )
                )
                print(f"loading pre-trained checkpoint")

            opt = torch.optim.Adam(model.parameters(), lr=LR_, weight_decay=WEIGHT_DECAY)

            scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=True)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader  = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

            for epoch in range(num_epochs+1):

                tr_loss = train_epoch(tr_loader, epoch, 
                            model=model, opt=opt)
                tr_loss_lst.append(tr_loss)

                if epoch%5==0:
                    val_loss = eval(val_loader, epoch, model=model)
                    val_loss_lst.append(val_loss)
                    scheduler.step(val_loss)

                if epoch%50==0: 

                    torch.save(model.state_dict(), model_dir+f"ap2_4l_%d_ep%d.pt"%(fold, epoch))

                    np.save("check/loss.npy", {'tr_loss':tr_loss_lst, 'val_loss':val_loss_lst})