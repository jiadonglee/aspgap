# import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch.utils.data import DataLoader
from model import Spec2label
from data import GaiaXP_55coefs_alpha


if __name__ == "__main__":
    #=========================Data loading ================================
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_wise_xpcont_cut.npy"

    device = torch.device('cuda:0')
    TOTAL_NUM = 6000
    BATCH_SIZE = 1024

    gdata  = GaiaXP_55coefs_alpha(
        data_dir+tr_file, 
        total_num=TOTAL_NUM, 
        part_train=False, 
        device=device
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
    INPUT_LEN = 55*2+3

    model = Spec2label(
        n_encoder_inputs=INPUT_LEN,
        n_outputs=1, channels=512, n_heads=8, n_layers=8,
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
    itr = 1
    num_iters=100

    tr_select = "A"
    model_dir = "/data/jdli/gaia/model/1202_alpha/" + tr_select

    if tr_select=="A":
        tr_loader = A_loader
        # check_point = model_dir +"/sp2_4labels_mse_A_ep0.pt"

    elif tr_select=="B":
        tr_loader = B_loader
        # check_point = "/data/jdli/gaia/model/1119/B/sp2_4labels_mse_B_ep85.pt"
    print("===================================")
    # print("Loading checkpoint %s"%(check_point))
    # model.load_state_dict(torch.load(check_point))

    print("Traing %s begin"%tr_select)

    def train_epoch(tr_loader, epoch):
        # model.train()
        model.train()
        total_loss = 0.
        start_time = time.time()

        for batch, data in enumerate(tr_loader):

            output = model(data['x']).view(-1,1)
            # loss = cost(output, data['y'], data['e_y'])
            loss = cost(output, data['y'])
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss+=loss_value
            del data, output

        print("epoch %d train loss:%.4f | %.4f s"%(epoch, total_loss/(batch+1e-5), time.time()-start_time))
    
    
    def eval(val_loader):
        model.eval()
        total_val_loss=0
        with torch.no_grad():
            for bs, data in enumerate(val_loader):

                output = model(data['x']).view(-1,1)
                # loss = cost(output.view(-1,1), data['y'], data['e_y'])
                loss = cost(output, data['y'])
                total_val_loss+=loss.item()
                del data, output

        print("val loss:%.4f"%(total_val_loss/(bs+1e-5)))

    num_epochs = 200
    for epoch in range(num_epochs+1):
        train_epoch(tr_loader, epoch)

        if epoch%5==0:
            eval(val_loader)
        if epoch%50==0: 
            save_point =  "/sp2_alpha_mse_%s_ep%d.pt"%(tr_select, epoch)
            torch.save(model.state_dict(), model_dir+save_point)

    torch.cuda.empty_cache()