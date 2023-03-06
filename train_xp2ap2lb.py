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
- specformer model script
"""

import time
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from kvxp.apxp import MLP, MLP_up, simpMLP, CNN
from kvxp.data import GXP_AP_4lb
from kvxp.utils import *
from torch import Tensor, nn

##==================Model==
#======================================

def train_epoch(tr_loader, epoch, model1=None, model2=None, opt1=None, opt2=None):

    model2.train()
    total_loss = 0.0
    start_time = time.time()
    
    itr=0
    for batch, data in enumerate(tr_loader):

        xp = data['x'][:, :n_xp]
        ap = data['x'][:, n_xp:]
        # 
        ap_pred = model2(xp)
        y2 = model1(ap_pred)
        loss2 = cost_mse(
            y2.view(-1,4), 
            data['y'].view(-1,4)
            )
        
        opt1.zero_grad()
        opt1.step()

        opt2.zero_grad()
        opt2.step()
        loss2.backward()
        total_loss+=loss2.item()
        itr+=1

    print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
    

def eval(val_loader, epoch, model1=None, model2=None, opt1=None, opt2=None):
    model1.eval()
    model2.eval()
    # model_pre.eval()
    total_val_loss=0
    itr=0
    # with torch.no_grad():
    for batch, data in enumerate(val_loader):

        xp = data['x'][:, :n_xp]
        y2 = model1(model2(xp))

        loss2 = cost_mse(
            y2.view(-1,4), 
            data['y'].view(-1,4)
            )

        loss_value = loss2.item()
        total_val_loss+=loss_value
        itr+=1

    print("val loss:%.4f"%(total_val_loss/itr))
    return total_val_loss/itr


#==============================
#       Hyper parameters
#==============================

"""
traing params
"""
band = "xp"
mask_band = "ap"
device = torch.device('cuda:0')
BATCH_SIZE = int(2**10)
num_epochs = 500
part_train = False

"""
data params
"""
data_dir = "/data/jdli/gaia/"
tr_file = "ap_xp_13286.npy"
"""
model params
"""

n_xp = 110
n_ap = 7514
n_labels = 4
# LR = 5e-5
LR_ = 5e-4
LR_xp2ap = 5e-4
loss_penal = 0.1
LMBDA_PEN = 1e-10
LMBDA_ERR = 1e-1

model_dir = f"/data/jdli/gaia/model/0303_ap2xp/"
# save_preflix = f"sp2_4l_%d_ep%d.pt"
# Check if the directory exists

if not os.path.exists(model_dir):
# Create the directory
    print("make dir %s"%model_dir)
    os.makedirs(model_dir)
else:
    print(f"save trained-model to {model_dir}")
#=========================
#        Data loading ===========================

gdata  = GXP_AP_4lb(
    data_dir+tr_file,
    part_train=part_train, 
    device=device,
)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

#==========================================
print("Training Start")     

for fold, (train_ids, valid_ids) in enumerate(kfold.split(gdata)):
    
    print(f'FOLD {fold}')
    print('--------------------------------')

    if fold==0:

        model_ap2l = CNN(n_ap, n_labels).to(device)
        # model_name = "/data/jdli/gaia/model/0220/"+"ap2_4l_%d_ep%d.pt"%(0, 300)
        # pre_dict = remove_prefix(torch.load(model_name))
        # print(f"Loading pre-trained model: {model_name}")
        # model_pre.load_state_dict(pre_dict)

        # Freeze all layers except the last one
        # for name, param in model_pre.named_parameters():
        #     param.requires_grad = False

        model_xp2ap = CNN(n_xp, n_ap).to(device)

        optimizer_xp2ap = torch.optim.Adam(
            model_xp2ap.parameters(), 
            lr=LR_xp2ap, weight_decay=1e-8
            )

        optimizer_ap2l = torch.optim.Adam(
            model_xp2ap.parameters(), 
            lr=LR_, weight_decay=1e-8
            )
        
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_xp2ap, 'min', patience=10, verbose=True
            )
        
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        
        tr_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

        for epoch in range(num_epochs+1):

            # if epoch>100:
            #     optimizer_com = None
            # else:
            #     optimizer_com = None

            train_epoch(
                tr_loader, epoch, 
                model1=model_ap2l, model2=model_xp2ap, 
                opt1=optimizer_ap2l, opt2=optimizer_xp2ap,
                )

            if epoch%5==0:
                val_loss = eval(
                val_loader, epoch, 
                model1=model_ap2l, model2=model_xp2ap, 
                opt1=optimizer_ap2l, opt2=optimizer_xp2ap,
                )
                # scheduler.step(val_loss)

            if epoch%100==0: 
                torch.save(
                    model_ap2l.state_dict(), model_dir+f"ap2_4l_%d_ep%d.pt"%(fold, epoch)
                    )
                torch.save(
                    model_xp2ap.state_dict(), model_dir+f"xp2ap_%d_ep%d.pt"%(fold, epoch)
                    )