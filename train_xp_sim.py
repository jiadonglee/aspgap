import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch.utils.data import DataLoader
from model import Spec2HRd, Spec2label
from data import GaiaXPlabel_cont


if __name__ == "__main__":
    #=========================Data loading ================================
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_xpcont_cut.npy"

    device = torch.device('cuda:0')
    TOTAL_NUM = 5000
    BATCH_SIZE = 1024

    gdata  = GaiaXPlabel_cont(data_dir+tr_file, total_num=TOTAL_NUM,
    part_train=True, device=device)

    val_size = int(0.1*len(gdata))
    A_size = int(0.5*(len(gdata)-val_size))
    B_size = len(gdata) - A_size - val_size

    A_dataset, B_dataset, val_dataset = torch.utils.data.random_split(gdata, [A_size, B_size, val_size], generator=torch.Generator().manual_seed(42))
    print(len(A_dataset), len(B_dataset), len(val_dataset))

    A_loader = DataLoader(A_dataset, batch_size=BATCH_SIZE)
    B_loader = DataLoader(B_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=2048)

    ##==================Model parameters============================
    ##==============================================================
    #===============================================================
    model = Spec2label(
        n_encoder_inputs=8+8+3, n_outputs=4, 
        channels=512, n_heads=8, n_layers=8,
    ).to(device)

    # cost = torch.nn.GaussianNLLLoss(full=True, reduction='mean')
    cost = torch.nn.MSELoss(reduction='mean')
    LRATE = 1e-5
    optimizer =torch.optim.Adam(model.parameters(), lr=LRATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
            )
    
    itr = 1
    num_iters  = 100
    # log_dir   = "/data/jdli/gaia/model/forcasting_1110A.log"
    model_dir = "/data/jdli/gaia/model/1116/"

    tr_select = "A"

    if tr_select=="A":
        tr_loader = A_loader
        # check_point = "sp2_4lbs_mse_A_ep100.pt"
        #     print("Loading model checkpoint %s"%(check_point))
    # model.load_state_dict(torch.load(model_dir+check_point)) 

    elif tr_select=="B":
        tr_loader = B_loader
        # check_point = "model/1115/enc_GXPcont_221110_NGlll_B_ep300.pt"        

    num_epochs = 200
    print("Traing %s begin"%tr_select)

    def train_epoch(tr_loader):
        # model.train()
        model.train()
        # model_mohaom.train()
        total_loss = 0.
        start = time.time()
        for batch, data in enumerate(tr_loader):

            output = model(data['x']).view(-1,4)
            # mohaom = model_mohaom(torch.concat((data['x'], tefflogg.view(-1, 1, 2)), dim=2))
            # loss = cost(output, data['y'], data['e_y'])
            # output = torch.concat((tefflogg, mohaom), dim=1)
            # loss = cost(tefflogg, data['y'][:,:2],)
            # loss = cost(output, data['y'], data['e_y'])
            loss = cost(output, data['y'])
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_loss+=loss_value
            # del data, output

        end = time.time()
        print(f"Epoch #%d tr loss:%.4f time:%.2f s"%(epoch, total_loss/(batch+1), (end-start)))

    
    def eval(val_loader):
        model.eval()
        # model_mohaom.eval()
        with torch.no_grad():
            total_val_loss = 0
            for bs, data in enumerate(val_loader):
                # output = model(data['x'])
                output = model(data['x'])
                # loss = cost(output, data['y'], data['e_y'])
                # output = torch.concat((tefflogg, mohaom), dim=1)
                loss = cost(output, data['y'])
                total_val_loss+=loss.item()
                # del data, output
        print("val loss:%.4f"%(total_val_loss/(bs+1)))


    for epoch in range(num_epochs+1):
        train_epoch(tr_loader)

        if epoch%5==0:
            eval(val_loader)

        if epoch%50==0: 
            save_point_tefflogg =  "sp2_4lbs_mse_%s_ep%d.pt"%(tr_select, epoch)
            # save_point_mohaom   =  "sp2mohaom_mse_%s_ep%d.pt"%(tr_select, epoch)

            torch.save(model.state_dict(), model_dir+save_point_tefflogg)
            # torch.save(model_mohaom.state_dict(), model_dir+save_point_mohaom)