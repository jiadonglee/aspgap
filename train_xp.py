# import numpy as np
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
    tr_file = "ap17_xpcont_trsnr300.npy"

    device = torch.device('cuda:0')
    TOTAL_NUM = 6000
    BATCH_SIZE = 1024

    gdata  = GaiaXPlabel_cont(data_dir+tr_file, total_num=TOTAL_NUM,
    part_train=False, device=device)

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
    model_tefflogg = Spec2label(
        n_encoder_inputs=8+8+3, n_outputs=2, 
        channels=128, n_heads=8, n_layers=8,
    ).to(device)

    model_mohaom = Spec2label(
        n_encoder_inputs=8+8+3+2, n_outputs=2, 
        channels=128, n_heads=8, n_layers=8,
    ).to(device)

    # model = Spec2label(
    #     n_encoder_inputs=8+8+3, n_decoder_inputs=8+8+3+2, n_outputs=2, channels=345, nhead=3
    #     ).to(device)

    # cost = torch.nn.GaussianNLLLoss(full=True, reduction='sum')
    cost = torch.nn.MSELoss(reduction='mean')

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(list(model_tefflogg.parameters()) + list(model_mohaom.parameters()), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
            )
    
    # num_batches = train_size//BATCH_SIZE
    itr = 1
    num_iters=100

    # log_dir   = "/data/jdli/gaia/model/forcasting_1110A.log"
    model_dir = "/data/jdli/gaia/model/1116/"
    
    tr_select = "B"

    if tr_select=="A":
        tr_loader = A_loader
        # check_point = "model/1115/enc_GXPcont_221110_NGlll_A_ep500.pt" 
        # check_point = "sp2hrd_mse_A_ep15.pt"

    elif tr_select=="B":
        tr_loader = B_loader
        # check_point = "model/1115/enc_GXPcont_221110_NGlll_B_ep300.pt"

    # print("Loading checkpint %s"%(check_point))
    # model.load_state_dict(torch.load(model_dir+check_point))

    num_epochs = 500
    print("Traing %s begin"%tr_select)

    def train_epoch(tr_loader):
        # model.train()
        model_tefflogg.train()
        model_mohaom.train()

        start = time.time()
        for batch, data in enumerate(tr_loader):
            total_loss = 0.

            # output = model(data['x'], data['y'])
            tefflogg = model_tefflogg(data['x'])
            mohaom = model_mohaom(torch.concat((data['x'], tefflogg.view(-1, 1, 2)), dim=2))
            # loss = smape_loss(output.view(-1), data['y'].view(-1))
            # loss = cost(output, data['y'], data['e_y'])

            output = torch.concat((tefflogg, mohaom), dim=1)
            loss = cost(output.view(-1, 4), data['y'],)
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_tefflogg.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model_mohaom.parameters(), 0.5)
            optimizer.step()

            total_loss += loss_value
            del data, output

            # if itr%num_iters == 0:
        end = time.time()

        print(f"Epoch #%d tr loss:%.4f time:%.2f s"%(epoch, total_loss/(batch+1), (end-start)))

    
    def eval(val_loader):
        model_tefflogg.eval()
        model_mohaom.eval()
        with torch.no_grad():
            total_val_loss = 0
            for bs, data in enumerate(val_loader):

                tefflogg = model_tefflogg(data['x'])
                mohaom = model_mohaom(torch.concat((data['x'], tefflogg.view(-1, 1, 2)), dim=2))
                # loss = smape_loss(output.view(-1), data['y'].view(-1))
                # loss = cost(output, data['y'], data['e_y'])

                output = torch.concat((tefflogg, mohaom), dim=1)
                loss = cost(output.view(-1, 4), data['y'],)
                total_val_loss+=loss.item()
                del data, output

        print("val loss:%.4f"%(total_val_loss/(bs+1)))


    for epoch in range(num_epochs+1):
        train_epoch(tr_loader)

        if epoch%5==0:
            eval(val_loader)

        if epoch%50==0: 
            save_point_tefflogg =  "sp2tefflogg_mse_%s_ep%d.pt"%(tr_select, epoch)
            save_point_mohaom   =  "sp2mohaom_mse_%s_ep%d.pt"%(tr_select, epoch)

            torch.save(model_tefflogg.state_dict(), model_dir+save_point_tefflogg)
            torch.save(model_mohaom.state_dict(), model_dir+save_point_mohaom)
            