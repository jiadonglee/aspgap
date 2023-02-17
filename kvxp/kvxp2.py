# -*- coding: utf-8 -*-
"""
Author
------
JDL
Email
-----
jiadong.li at nyu.edu
Created on
----------
- Fri Jan 31 12:00:00 2023
Modifications
-------------
- Fri Feb 14 12:00:00 2023
Aims
----
- specformer model script
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=1000):
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

        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = self.position_embedding(x)+self.value_embedding(x)
        x = self.value_embedding(x)
        return self.dropout(x)



class specformer2(nn.Module):
    def __init__(self, input_size, n_outputs, n_hi=7514, hidden_size=128, channels=32, concat_size=388, num_heads=4, num_layers=4, dropout=0.1):
        super(specformer2, self).__init__()
        self.n_enc = input_size
        self.n_hi = n_hi
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(concat_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout,
            dim_feedforward=2*channels, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout,
            dim_feedforward=2*channels, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.conv0 = nn.Conv1d(1, channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, stride=3, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=3, padding=0)
        self.fc1 = nn.Linear(channels*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_outputs)
        self.do = nn.Dropout(p=dropout)


    def forward(self, x, mask=None, inf=False):
        src = x.view(-1, 1, self.n_enc+self.n_hi)
        x0 = src[:,:,:self.n_enc]
        src_x = F.leaky_relu(self.conv0(x0)) #(bs, dim, 108)

        if inf:
            src = src_x
        else:
            src_a = src[:,:,-self.n_hi:]
            src_a = F.leaky_relu(self.conv1(src_a))
            src_a = F.leaky_relu(self.conv2(src_a))
             # (bs, dim, 278)
            src_a = F.leaky_relu(self.conv2(src_a))
            src = torch.concat((src_x, src_a), 2) # (bs, dim, 108+278)
        
        # x = self.layer_norm(src).permute(0,2,1) # (bs, 110+278, dim)
        # # apply multi-head self-attention
        # attn_output, _ = self.self_attn(x,x,x, key_padding_mask=mask)
        # x = x + attn_output
        # x = self.do(x)
        src = src.permute(0, 2, 1) # (bs, 110(+278), dim)
        src = self.encoder(src, src_key_padding_mask=mask)
        
        x = self.decoder(tgt=src_x.permute(0, 2, 1), memory=src)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # apply feed-forward layers
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to perform inference on the model
def inference(model, input_data, mask_index, device):

    bs, n_enc = input_data.size(0), 386
    # Create a mask with 1s at the positions to be masked
    mask = torch.zeros([bs, n_enc], dtype=torch.bool, device=device)
    mask[:, mask_index] = True
    # Pass the input through the model
    output = model(input_data, mask)
    return output
        
        
class CNN(nn.Module):
    def __init__(self, n_input=7514, n_output=4, in_channel=1):
        super().__init__()
        self.n_input = n_input
        self.in_channel=in_channel
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * n_input, 128)
        self.fc2 = nn.Linear(128, n_output)
        
    def forward(self, x):
        # print(x.shape)
        x = x.permute(0,2,1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


