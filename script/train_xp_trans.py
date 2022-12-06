import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from transformer import TransformerReg, generate_square_subsequent_mask
from data import GaiaXPlabel_forcast


def infer(model: nn.Module, src: torch.Tensor, forecast_window:int, device) -> torch.Tensor:
    
    target_seq_dim = 1
    tgt = src[:,-1,0].view(-1,1,1) # [bs, 1, 1]

    for _ in range(forecast_window-1):
        dim_a = tgt.shape[1] #1,2,3,.. n
        dim_b = src.shape[1] #30
        
        src_mask = generate_square_subsequent_mask(dim1=dim_a, dim2=dim_b).to(device)
        tgt_mask = generate_square_subsequent_mask(dim1=dim_a, dim2=dim_a).to(device)
        
        prediction = model(src, tgt, src_mask, tgt_mask)
        # Obtain the predicted value at t+1 where t is the last step 
        # represented in tgt
        last_predicted_value = prediction[:,-1,:].view(-1,1,1) #[bs, 1]
        # print(tgt.size())
        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value), dim=target_seq_dim)
    
    src_mask = generate_square_subsequent_mask(dim1=4, dim2=30).to(device)
    tgt_mask = generate_square_subsequent_mask(dim1=4, dim2=4).to(device)
    # Make final prediction
    return model(src, tgt, src_mask, tgt_mask)


if __name__ == "__main__":
    #=========================Data loading ================================
    data_dir = "/data/jdli/gaia/"
    tr_file = "ap17_xpcont_trsnr300.npy"

    device = torch.device('cuda:0')
    TOTAL_NUM = 5000
    BATCH_SIZE = 1024

    gdata  = GaiaXPlabel_forcast(data_dir+tr_file, total_num=TOTAL_NUM,
    part_train=True, device=device)

    val_size = int(0.1*len(gdata))
    A_size = int(0.5*(len(gdata)-val_size))
    B_size = len(gdata) - A_size - val_size

    A_dataset, B_dataset, val_dataset = torch.utils.data.random_split(gdata, [A_size, B_size, val_size], generator=torch.Generator().manual_seed(42))
    print(len(A_dataset), len(B_dataset), len(val_dataset))

    A_loader = DataLoader(A_dataset, batch_size=BATCH_SIZE)
    B_loader = DataLoader(B_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=2048)


    src_mask = generate_square_subsequent_mask(dim1=4, dim2=30).to(device)
    tgt_mask = generate_square_subsequent_mask(dim1=4, dim2=4).to(device)

    ##==================Model parameters============================
    ##==============================================================
    #===============================================================
    enc_seq_len = 30
    out_seq_len = 4

    model = TransformerReg(
        dim_val=128, input_size=1, batch_first=True, 
        enc_seq_len=enc_seq_len, 
        dec_seq_len=out_seq_len, out_seq_len=out_seq_len, 
        n_decoder_layers=2, n_encoder_layers=2, 
        n_heads=4, max_seq_len=30,
    ).to(device)

    src_mask = generate_square_subsequent_mask(
    dim1=4, dim2=30
    ).to(device)

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = generate_square_subsequent_mask( 
        dim1=4, dim2=4
        ).to(device)

    # cost = torch.nn.GaussianNLLLosss(full=True, reduction='mean')
    cost = torch.nn.MSELoss(reduction='mean')
    LRATE = 1e-5
    optimizer =torch.optim.Adam(model.parameters(), lr=LRATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
            )
    
    itr = 1
    num_iters  = 100

    tr_select = "A"
    model_dir = "/data/jdli/gaia/model/1119/" + tr_select

    if tr_select=="A":
        tr_loader = A_loader
        
    elif tr_select=="B":
        tr_loader = B_loader
      

    num_epochs = 50
    print("Traing %s begin"%tr_select)

    def train_epoch(tr_loader):
        # model.train()
        model.train()
        # model_mohaom.train()
        total_loss = 0.
        start = time.time()
        for batch, data in enumerate(tr_loader):

            output = model(data['x'].view(-1,enc_seq_len,1), 
            data['y'].view(-1,out_seq_len,1), src_mask, tgt_mask)
            loss = cost(output, data['y'].view(-1,out_seq_len,1))
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_loss+=loss_value
            del data, output

        end = time.time()
        print(f"Epoch #%d tr loss:%.4f time:%.2f s"%(epoch, total_loss/(batch+1), (end-start)))

    
    def eval(val_loader):
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for bs, data in enumerate(val_loader):
                # output = model(data['x'], data['tgt'])
                output = infer(model=model, src=data['x'].view(-1,enc_seq_len,1), forecast_window=4, device=device)

                loss = cost(output, data['y'].view(-1,out_seq_len,1))
                total_val_loss+=loss.item()

        print("val loss:%.4f"%(total_val_loss/(bs+1)))


    for epoch in range(num_epochs+1):
        train_epoch(tr_loader)

        # if epoch%5==0:
        #     eval(val_loader)

        if epoch%10==0: 
            save_point =  "sp2_trans_mse_%s_ep%d.pt"%(tr_select, epoch)

            torch.save(model.state_dict(), model_dir+save_point)
