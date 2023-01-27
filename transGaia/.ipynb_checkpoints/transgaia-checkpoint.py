import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        # x = x.permute(1,0,2)
        pos = self.pe[:x.size(1)]
        return self.dropout(pos).permute(1,0,2)



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.position_embedding(x)+self.value_embedding(x)
        return self.dropout(x)


class Spec2label(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, channels=128, n_heads=8, n_layers=8, dropout=0.2, attn=False):
        super().__init__()

        # self.save_hyperparameters()
        self.n_encoder_inputs = n_encoder_inputs
        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.attn = attn
        self.enc_embedding = DataEmbedding(1, d_model=channels, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dropout=self.dropout,
            dim_feedforward=4*channels, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(n_encoder_inputs, channels)

        self.fc1 = nn.Linear(channels*n_encoder_inputs, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_outputs)
        self.do  = nn.Dropout(p=self.dropout)


    def forward(self, x):
        src = self.enc_embedding(x.view(-1, self.n_encoder_inputs, 1)) # (bs, n_enc, dim)
        src = self.encoder(src) # (bs, n_enc, dim)
        src_enc = src
        
        if self.attn:
            attention_maps = []
            for l in self.encoder.layers:
                attention_maps.append(l.self_attn(src, src, src)[1])

        # src = src.view(-1, self.channels*self.n_encoder_inputs) # (bs, 512*113)
        src = torch.flatten(src, start_dim=1) #(bs, 512*113)
        src = F.relu(self.fc1(src)) # (bs, 1024)
        src = F.relu(self.fc2(src)) # (bs, 128)
        src = F.relu(self.fc3(src)) # (bs, 1)
        tgt = self.do(src)

        if self.attn:
            return tgt, attention_maps, src_enc
        else:
            return tgt




class xp2label(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, channels=64,
        n_heads=8, n_layers=8, dropout=0.2, attn=False):
        super().__init__()

        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.input_pos_embedding  = torch.nn.Embedding(128, embedding_dim=channels)
        self.attn = attn
        self.n_enc = n_encoder_inputs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dropout=self.dropout, dim_feedforward=4*channels, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(1, channels)
        # self.output_projection = Linear(n_decoder_inputs, channels)
        # self.linear = Linear(channels, 2)
        self.fc1 = nn.Linear(channels*n_encoder_inputs, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, n_outputs)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)
        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0).repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)
        src = src_start + pos_encoder

        if self.attn:
            attention_maps = []
            for l in self.encoder.layers:
                attention_maps.append(l.self_attn(src, src, src)[1])
            src = self.encoder(src)
            return src, attention_maps
        else:
            src = self.encoder(src)
            return src

    def forward(self, x):
        src = x.view(-1, self.n_enc, 1)
        if self.attn:
            src, attention_maps = self.encode_src(src) # (1, bs, 512)
        else:
            src = self.encode_src(src)
            
        src_enc = src
    
        src = torch.flatten(src, start_dim=1)
        src = self.do(src)
        src = F.relu(self.fc1(src))
        src = F.relu(self.fc2(src))
        tgt = F.relu(self.fc3(src)) # (bs, 1)
        
        if self.attn:
            return tgt, attention_maps, src_enc
        else:
            return tgt