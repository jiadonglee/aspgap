# from cmath import log
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch.utils.data import DataLoader
from model import Spec2label, smape_loss
from data import GaiaXPlabel_v2


if __name__ == "__main__":
    #=========================Data loading ================================
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_xp.npy"

    device = torch.device('cuda:0')
    TOTAL_NUM = 6000
    BATCH_SIZE = 16

    gdata  = GaiaXPlabel_v2(data_dir+tr_file, total_num=TOTAL_NUM,
    part_train=False, device=device)

    val_size = int(0.1*len(gdata))
    A_size = int(0.5*(len(gdata)-val_size))
    B_size = len(gdata) - A_size - val_size

    A_dataset, B_dataset, val_dataset = torch.utils.data.random_split(gdata, [A_size, B_size, val_size], generator=torch.Generator().manual_seed(42))
    print(len(A_dataset), len(B_dataset), len(val_dataset))

    A_loader = DataLoader(A_dataset, batch_size=BATCH_SIZE, )
    B_loader = DataLoader(B_dataset, batch_size=BATCH_SIZE, )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    ##==================Model parameters============================
    ##==============================================================
    #===============================================================
    model = Spec2label(
        n_encoder_inputs=343,
        n_outputs=2,
        lr=1e-5,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
    
    total_loss = 0.
    num_epochs = 100
    # num_batches = train_size//BATCH_SIZE
    itr = 1
    num_iters  = 100

    tr_select = "B"

    if tr_select=="A":
        tr_loader = A_loader
    elif tr_select=="B":
        tr_loader = B_loader


    log_dir   = "/data/jdli/gaia/model/forcasting_1107A.log"
    model_dir = "/data/jdli/gaia/model/"
    save_model_name = "model/enc_GXP_221109"+tr_select+"ep"+str(num_epochs)+".pt"

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )
    
    print("Traing %s begin"%tr_select)

    for epoch in range(num_epochs):
        model.train()
        
        for batch, data in enumerate(A_loader):
            start = time.time()

            # output = model(data['x'], data['y'])
            output = model(data['x'])
            loss = smape_loss(output.view(-1), data['y'].view(-1))
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            del data, output

            if itr%num_iters == 0:
                end = time.time()
                print(f"Epoch #%d  Iteration #%d  tr loss:%.4f time:%.2f s"%(epoch, itr, total_loss/itr, (end-start)*num_iters))
                    # writer.add_scalar('training loss = ',loss_value,epoch*itr)
                model.eval()
                with torch.no_grad():
                    k=0
                    total_val_loss = 0
                    for data in val_loader:
                        output = model(data['x'])
                        loss = smape_loss(output.view(-1), data['y'].view(-1))
                        total_val_loss+=loss.item()
                        k+=1
                        del data, output
                print("val loss:%.4f"%(total_val_loss/k))

            itr+=1

    torch.save(model.state_dict(), data_dir+save_model_name)