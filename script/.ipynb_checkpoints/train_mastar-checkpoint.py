import astropy.io.fits as fits
import numpy as np
# %load_ext autoreload
# %autoreload 2
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformer import TransformerReg

class Mastar():
    """ 
    MaStar spectra instance
    """

    def __init__(self, pars, specs, device=torch.device('cuda:0')):
        self.waves, self.fluxes = specs["WAVE"], specs["FLUX"]
        self.device = device
        self.MANGAID  = np.array(pars["MANGAID"])
        self.MANGAID_specs = np.array(specs["MANGAID"])
        self.pars = np.c_[pars['BPRPC'], pars['M_G']]
        
    def __len__(self) -> int:
        num_sets = len(self.MANGAID)
        return num_sets
    
    def __getitem__(self, idx: int):
        mangaid = self.MANGAID[idx]
        visit   = self.MANGAID_specs==mangaid
        flux    = torch.tensor(self.fluxes[visit][0].reshape(-1,1).astype(np.float32))
        output  = torch.tensor(self.pars[idx].reshape(-1,1).astype(np.float32))
        return flux.to(self.device), output.to(self.device)


if __name__ == "__main__":

    data_dir = "/data/jdli/mastar/"
    photom = fits.open(data_dir + 'mastarall-gaiaedr3-extcorr-simbad-ps1-v3_1_1-v1_7_7-v1.fits')[1].data
    clean_match = photom['GAIA_CLEANMATCH'] #mask of those entries with a clean match in Gaia
    mask = np.where((photom['BPRPC']>-999.)&(photom['GAIA_CLEANMATCH']==1.))

    goodspec = np.load(data_dir+"mastar-goodspec-v3_1_1-v1_7_7.npy", allow_pickle=True).item()

    device = torch.device('cuda:1')
    TOTAL_SIZE = 6000
    BATCH_SIZE = 8

    ## Model parameters
    dim_val = 4563 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 9 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 2 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 2 # Number of times the encoder layer is stacked in the encoder
    input_size = 1 # The number of input variables. 1 if univariate forecasting.
    enc_seq_len = 4563 # length of input given to encoder. Can have any integer value.
    dec_seq_len = 2 # length of input given to decoder. Can have any integer value.
    output_sequence_length = 2 # Length of the target sequence, i.e. how many time steps should your forecast
    # max_seq_len = 5000 # What's the longest sequence the model will encounter? Used to make the positional encoder

    device = torch.device('cuda:1')

    model = TransformerReg(dim_val=dim_val, input_size=input_size, 
                           batch_first=True, dec_seq_len=dec_seq_len, 
                           out_seq_len=output_sequence_length, n_decoder_layers=n_decoder_layers,
                           n_encoder_layers=n_encoder_layers, n_heads=n_heads,
                           max_seq_len=8,
                           ).to(device)


    mastar = Mastar(photom[mask][:TOTAL_SIZE], goodspec, device=torch.device('cuda:1'))

    train_size = int(0.75*len(mastar))
    val_size = len(mastar) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mastar, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))


    tr_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, )
    val_loader = DataLoader(val_dataset,  batch_size=BATCH_SIZE, )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.
    num_epochs = 30
    num_batches = train_size//BATCH_SIZE
    itr = 1
    num_iters  = 50

    for epoch in range(num_epochs):
        model.train()
        for batch, (x, y) in enumerate(tr_loader):
            start = time.time()
            
            output = model(x, y)
            loss = criterion(output, y)
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            del x, y, output
            
            if itr%num_iters == 0:
                end = time.time()
                print(f"Epoch #%d  Iteration #%d  tr loss:%.4f time:%.2f s"%(epoch, itr, total_loss/itr, (end-start)*num_iters))
                    # writer.add_scalar('training loss = ',loss_value,epoch*itr)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    k=0
                    for x, y in val_loader:
                        output = model(x, y)
                        loss = criterion(output, y)
                        total_val_loss += loss.item()
                        k+=1
                        del x, y, output
                print("val loss:%.4f"%(total_val_loss/k))
            itr+=1


    torch.save(model.state_dict(), data_dir+"trsfm_221015.pt")