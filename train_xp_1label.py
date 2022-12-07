# import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch.utils.data import DataLoader
from transGaia.transgaia import xp2label
from transGaia.data import GXP_5lb



if __name__ == "__main__":
    #======================== Hyper parameters=============================
    """traing params
    """
    device = torch.device('cuda:1')
    TOTAL_NUM = 6000
    BATCH_SIZE = 512
    num_epochs = 200

    """data params
    """
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_wise_xp_70123.npy"

    """model params
    """
    INPUT_LEN = 55*2+3
    tr_select = "A"


    label_names = ["Teff", "Logg",  "PRLX", "M_H", "ALPHA_M"]
    select_1l = "ALPHA_M"
    idx_1l = label_names.index(select_1l)

    
    part_train = False

    #=========================Data loading ================================

    gdata  = GXP_5lb(
        data_dir+tr_file, total_num=TOTAL_NUM, 
        part_train=part_train,  device=device,
    )

    val_size = int(0.1*len(gdata))
    A_size = int(0.5*(len(gdata)-val_size))
    B_size = len(gdata) - A_size - val_size

    A_dataset, B_dataset, val_dataset = torch.utils.data.random_split(gdata, [A_size, B_size, val_size], generator=torch.Generator().manual_seed(42))

    print(len(A_dataset), len(B_dataset), len(val_dataset))

    A_loader = DataLoader(A_dataset, batch_size=BATCH_SIZE)
    B_loader = DataLoader(B_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    ##==================Model parameters============================
    ##==============================================================
    #===============================================================
    

    model = xp2label(
        n_encoder_inputs=INPUT_LEN,
        n_outputs=1, channels=128, n_heads=8, n_layers=8,
    ).to(device)

    # cost = torch.nn.GaussianNLLLoss(full=True, reduction='mean')
    cost = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5, weight_decay=1e-6
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    
    model_dir = "/data/jdli/gaia/model/1207_alpha/" + tr_select

    if tr_select=="A":
        tr_loader = A_loader
        # check_point = model_dir +"/sp2_teff_robustnorm_mse_A_ep150.pt"

    elif tr_select=="B":
        tr_loader = B_loader

    print("===================================")
    # print("Loading checkpoint %s"%(check_point))
    # model.load_state_dict(torch.load(check_point))

    print("Traing %s begin"%tr_select)
    print("Training label name = %s"%select_1l)

    def train_epoch(tr_loader, epoch):
            # model.train()
        model.train()
        total_loss = 0.
        start_time = time.time()
        
        itr=0
        for batch, data in enumerate(tr_loader):
            output = model(data['x'])
            loss = cost(output.view(-1,1), data['y'][:,idx_1l].view(-1,1))
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
            for bs, data in enumerate(val_loader):
                output = model(data['x'])
                # loss = cost(output, data['y'], data['e_y'])
                loss = cost(output.view(-1,1), data['y'][:,idx_1l].view(-1,1))
                total_val_loss+=loss.item()
                del data, output
                itr+=1
        print("val loss:%.4f"%(total_val_loss/itr))


    for epoch in range(num_epochs+1):
        train_epoch(tr_loader, epoch)

        if epoch%5==0:
            eval(val_loader)
        if epoch%50==0: 
            save_point =  "/sp2_%s_robustnorm_mse_%s_ep%d.pt"%(select_1l, tr_select, epoch)
            torch.save(model.state_dict(), model_dir+save_point)

    torch.cuda.empty_cache()