from cmath import log
import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch.utils.data import DataLoader
from transformer import TransformerReg, TransAm
from data import GaiaXPlabel


if __name__ == "__main__":
    #=========================Data loading ================================
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_xp.npy"

    device = torch.device('cuda:1')
    TOTAL_NUM = 6000
    BATCH_SIZE = 64

    gdata  = GaiaXPlabel(data_dir+tr_file, total_num=TOTAL_NUM,
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
    dim_val = 64 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 4 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 2 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 2 # Number of times the encoder layer is stacked in the encoder
    extd_size = 0

    input_size = 1 # The number of input variables. 1 if univariate forecasting.
    enc_seq_len = 343 # length of input given to encoder. Can have any integer value.
    dec_seq_len = 2 # length of input given to decoder. Can have any integer value.
    output_sequence_length = 2
    max_seq_len = 343 # What's the longest sequence the model will encounter? Used to make the positional encoder
    # model = TransformerReg(
    #     input_size,  batch_first=True,  dec_seq_len=2,
    #     dim_val=dim_val, enc_seq_len=enc_seq_len,
    #     out_seq_len=output_sequence_length, n_decoder_layers=n_decoder_layers,n_encoder_layers=n_encoder_layers, n_heads=n_heads,
    #     max_seq_len=max_seq_len,
    #     ).to(device)
    model = TransAm(feature_size=64, enc_len=343, tgt_len=2, num_layers=2, dropout=0.2).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    total_loss = 0.
    num_epochs = 50
    # num_batches = train_size//BATCH_SIZE
    itr = 1
    num_iters  = 100

    tr_select = "B"

    if tr_select=="A":
        tr_loader = A_loader
    elif tr_select=="B":
        tr_loader = B_loader
        
    save_model_name = "model/trsfm_GXP_221105"+tr_select+"ep"+str(num_epochs)+".pt"
    
    print("Traing %s begin"%tr_select)
    for epoch in range(num_epochs):
        model.train()
        
        for batch, data in enumerate(A_loader):
            start = time.time()

            # output = model(data['x'], data['y'])
            output = model(data['x'])
            loss = criterion(output, data['y'])
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
                        loss = criterion(output, data['y'])
                        total_val_loss+=loss.item()
                        k+=1
                        del data, output
                print("val loss:%.4f"%(total_val_loss/k))

            itr+=1

    torch.save(model.state_dict(), data_dir+save_model_name)