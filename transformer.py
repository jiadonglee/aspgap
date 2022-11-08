import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import copy


class PositionalEncoder(nn.Module):
    def __init__(self,  dropout: float=0.1, max_seq_len:int=5000, d_model:int=512, batch_first:bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        if batch_first:
            if d_model%2 == 0:
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
                pe = torch.zeros(1, max_seq_len, d_model)
                
            else:
                div_term = torch.exp(torch.arange(0, d_model+1, 2) * (-math.log(10000.0)/d_model))
                pe = torch.zeros(1, max_seq_len, d_model+1)

            pe[0,:,0::2] = torch.sin(position * div_term)
            pe[0,:,1::2] = torch.cos(position * div_term)

        else:
            if d_model%2 == 0:
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
                pe = torch.zeros(max_seq_len, 1, d_model)
            else:
                div_term = torch.exp(torch.arange(0, d_model+1, 2) * (-math.log(10000.0)/d_model))
                pe = torch.zeros(max_seq_len, 1, d_model+1)

            pe[:,0,0::2] = torch.sin(position * div_term)
            pe[:,0,1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:

            if self.d_model%2 == 0:
                x = x + self.pe[:,:x.size(self.x_dim),:]
            else:
                x  = x + self.pe[:,:x.size(self.x_dim),-1]
        else:
            if self.d_model%2 == 0:
                x = x + self.pe[:x.size(self.x_dim)]
            else:
                x  = x + self.pe[:x.size(self.x_dim)-1]
        return self.dropout(x)
    

# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class TransformerEncoder(nn.Module):
#     __constants__ = ['norm']
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         # self.layers = _get_clones(encoder_layer, num_layers)
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
#         output = src
#         weights = []
#         for mod in self.layers:
#             output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             weights.append(weight)
#         if self.norm is not None:
#             output = self.norm(output)
#         return output, weights  
def loglikelihood(output, y):
    y, output = y.reshape(-1,2), output.reshape(-1,2)
    prlx_g, e_prlx_g = y[:,0], y[:,1]
    prlx_t, e_prlx_t = output[:,0], output[:,1]
    
    s = torch.log(torch.square(e_prlx_g)+torch.square(e_prlx_t))
    return torch.mean(0.5*(prlx_g-prlx_t)**2 * torch.exp(-s) + 0.5*s)

def loglikelihood_ML(output, y):
    y, output = y.reshape(-1,2), output.reshape(-1,2)
    prlx_g, e_prlx_g = y[:,0], y[:,1]
    prlx_t, e_prlx_t = output[:,0], output[:,1]
    
    s = torch.log(torch.square(e_prlx_g)+torch.square(e_prlx_t))
    return torch.sum(0.5*(prlx_g-prlx_t)**2 * torch.exp(-s) + 0.5*s)


class TransformerReg(nn.Module):

    """
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    [2] Vaswani, A. et al. (2017)
    """

    def __init__(
        self,
        input_size: int, 
        batch_first: bool=True, out_seq_len:int=58,
        enc_seq_len:int=304, dec_seq_len: int=2,
        dim_val:int=512, n_encoder_layers:int=4, n_decoder_layers:int=4, n_heads:int=8,
        dropout_encoder: float=0.2, dropout_decoder: float=0.2, dropout_pos_enc: float=0.1, 
        num_predicted_features: int=1, max_seq_len: int=8 ,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        ): 
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer of the decoder
            num_predicted_features: int, the number of features you want to predict. Most of the time, this will be 1.
        """
        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.enc_seq_len = enc_seq_len
        self.dim_val = dim_val
        self.output_sequence_length = out_seq_len

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=dim_val)

        self.linear_mapping = nn.Linear(
            in_features=dim_val*input_size, out_features=num_predicted_features
            )
        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val, dropout=dropout_pos_enc, max_seq_len=max_seq_len
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, nhead=n_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout_encoder, batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads,
                                                   dim_feedforward=dim_feedforward_decoder,
                                                   dropout=dropout_decoder, batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor=None, src_mask: Tensor=None, tgt_mask: Tensor=None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        """
        src = self.encoder_input_layer(src) 
        src = self.positional_encoding_layer(src)
        src = self.encoder(src) # src shape: [batch_size, enc_seq_len, dim_val]
        decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        decoder_output = self.linear_mapping(decoder_output) # shape [batch_size, target seq len]
        return decoder_output


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]    


class TransAm(nn.Module):
    def __init__(self, feature_size=64, nheads=8, enc_len=343, tgt_len=2, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.enc_len = enc_len
        self.tgt_len = tgt_len
        self.src_mask = None
        self.nhead = nheads
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size, 1)
        self.linear_map = nn.Linear(enc_len, tgt_len)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        # print(output.shape)
        output = self.decoder(output)
        output = output.view(output.size(0), -1, self.enc_len)
        output = self.linear_map(output)
        return output.view(output.size(0), self.tgt_len, 1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

class TimeSeriesForcasting(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        channels=512,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = nn.Linear(n_encoder_inputs, channels)
        self.output_projection = nn.Linear(n_decoder_inputs, channels)

        self.linear = nn.Linear(channels, 2)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory):

        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src, trg = x

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_in, trg_out = batch

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }