# import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from transGaia.transgaia import xp2label
from transGaia.data import GXP_5lb

if __name__ == "__main__":
    #======================== Hyper parameters=============================
    """traing params
    """
    device = torch.device('cuda:1')
    TOTAL_NUM = 1000
    BATCH_SIZE = 512
    num_epochs = 100
    part_train = False

    """data params
    """
    data_dir = "/data/jdli/gaia/"
    # tr_file = "ap17_wise_xp_66701_allstand1224.npy"
    tr_file = "ap17_wise_xp_66701_allstand1225.npy"

    """model params
    """
    INPUT_LEN = 55*2+3
    n_outputs = 8
    n_dim = 128
    n_head = 8
    n_layer = 8
    LR = 5e-5
    LMBDA_PEN = 5e-7
    model_dir = "/data/jdli/gaia/model/1226_4l_err_pen/"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)

    #=========================Data loading ================================
    gdata  = GXP_5lb(
        data_dir+tr_file, total_num=TOTAL_NUM, 
        part_train=part_train,  device=device,
    )
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)


    def cost_penalty(pred, tgt, e_pred, e_tgt, model, lmbda=LMBDA_PEN, lmbda_2=1e-5):
        """
        Computes the loss between the predictions and targets, with a penalty term.
        """
        err_min, err_max = 1e-5, 200

        var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
        cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
        loss = cost(pred, tgt, var)
        # Add the penalty term
        penalty = lmbda * sum(abs(p).sum() for p in model.parameters())
        # pen_par = torch.clamp(pred-err_threshold, min=0).sum()
        pen_error_max =  lmbda_2*torch.clamp(e_pred-err_max, min=0).sum()
        # pen_error_min = -lmbda_2*torch.clamp(e_pred-err_min, max=0).sum()
        loss += penalty + pen_error_max
        return loss

    def cost_val(pred, tgt, e_pred, e_tgt):
        err_min, err_max = 1e-5, 200

        var = torch.sqrt(torch.square(e_pred)+torch.square(e_tgt))
        cost = torch.nn.GaussianNLLLoss(reduction='mean', eps=err_min)
        loss = cost(pred, tgt, var)
        return loss

    ##==================Model ======================================
    def train_epoch(tr_loader, epoch):
            # model.train()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        itr=0
        for batch, data in enumerate(tr_loader):
            output = model(data['x']).view(-1, 8)
            
            loss = cost_penalty(
                output[:,:4], data['y'].view(-1,4), 
                output[:,4:], data['e_y'].view(-1,4), 
                model
                )
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss+=loss_value
            itr+=1
            del data, output
        print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/itr, time.time()-start_time))
        

    def eval(val_loader):
        model.eval()
        total_val_loss=0
        itr=0
        with torch.no_grad():
            for batch, data in enumerate(val_loader):

                output = model(data['x']).view(-1, 8)
                loss = cost_val(
                output[:,:4], data['y'].view(-1,4), 
                output[:,4:], data['e_y'].view(-1,4),
                )
                total_val_loss+=loss.item()
                del data, output
                itr+=1

        print("val loss:%.4f"%(total_val_loss/itr))
#==================================================================

    print("Training Start :================")     

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(gdata)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        if fold==2:
            LR_ = 0.2*LR
        else:
            LR_ = LR

        if fold>=0:
            model = xp2label(
                n_encoder_inputs=INPUT_LEN, n_outputs=n_outputs, channels=n_dim, n_heads=n_head, n_layers=n_layer
                ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LR_, weight_decay=1e-8
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(gdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

            for epoch in range(num_epochs+1):
                train_epoch(tr_loader, epoch)
                if epoch%5==0:
                    eval(val_loader)
                if epoch%25==0: 
                    save_point = "/sp2_4l_allstand_nll_%d_ep%d.pt"%(fold, epoch)
                    torch.save(model.state_dict(), model_dir+save_point)
            torch.cuda.empty_cache()
            del model