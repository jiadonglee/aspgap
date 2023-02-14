# import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from kvxp.kvxp import xp2label_nn, xp2label_attn
from kvxp.data import GXP_prlx


def cost_mse(pred, tgt):
    cost = torch.nn.MSELoss(reduction='mean')
    return cost(pred, tgt)


if __name__ == "__main__":

    #==============================
    #========== Hyper parameters
    """traing params
    """
    device = torch.device('cuda:0')
    TOTAL_NUM = int(5e4)
    BATCH_SIZE = 2**12
    num_epochs = 500
    part_train = True

    """data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "xp_pseudolum_737640.npy"

    """model params
    """
    INPUT_LEN = 110
    n_outputs = 1
    n_dim = 128
    n_head = 2
    n_layer = 2

    LR_ = 1e-4
    LMBDA_PEN = 1e-10
    LMBDA_ERR = 1e-1
    model_dir = "/data/jdli/gaia/model/0202_prlx/"

    # Check if the directory exists
    if not os.path.exists(model_dir):
    # Create the directory
        print("make dir %s"%model_dir)
        os.makedirs(model_dir)

    #=========================Data loading ================================
    apdata  = GXP_prlx(
        data_dir+tr_file, 
        part_train=part_train, 
        total_num=TOTAL_NUM,
        device=device
        )
    k_folds = 5

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    ##==================Model ======================================
    def train_epoch(tr_loader, epoch):
            # model.train()
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        itr=0
        for batch, data in enumerate(tr_loader):

            output = model(data['x'])
            # print(output.size(), data['y'].size())
            loss = cost_mse(output, data['y'].view(-1,1))
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

                output = model(data['x'])
                loss = cost_mse(output, data['y'].view(-1,1))
                total_val_loss+=loss.item()

                del data, output
                itr+=1

        print("val loss:%.4f"%(total_val_loss/itr))
        return total_val_loss/itr
#==================================================================

    print("Training Start :================")     

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(apdata)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        if fold==0:

            model = xp2label_nn(
                n_encoder_inputs=INPUT_LEN, n_outputs=n_outputs, channels=n_dim,
                ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=LR_, 
                weight_decay=1e-7
            )
            scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            
            tr_loader = DataLoader(apdata, batch_size=BATCH_SIZE, sampler=train_subsampler)
            val_loader = DataLoader(apdata, batch_size=BATCH_SIZE, sampler=valid_subsampler)

            for epoch in range(num_epochs+1):
                train_epoch(tr_loader, epoch)

                if epoch%5==0:
                    val_loss = eval(val_loader)
                    scheduler.step(val_loss)
                    torch.save(model.state_dict(), model_dir+"val_temp.pt")

                if epoch%25==0: 
                    save_point = "sp2_ap_prlx_%d_ep%d.pt"%(fold, epoch)
                    torch.save(model.state_dict(), model_dir+save_point)
            # torch.cuda.empty_cache()
            del model