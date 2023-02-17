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
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpecFormer(nn.Module):
    def __init__(self, input_size, n_outputs, n_hi=234, concat_size=388, hidden_size=128, channels=32, num_heads=4, num_layers=4, dropout=0.1, device=torch.device('cpu')):

        super(SpecFormer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_dim = channels
        self.num_hi = n_hi
        self.device = device

        self.layer_norm = nn.LayerNorm(input_size+n_hi)
        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, 
        bias=False, dropout=dropout)
        self.fc1 = nn.Linear((input_size+n_hi)*channels, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, n_outputs, bias=False)

    def add_position_encoding(self, x):
        # Get the dimension of the data
        k = x.shape[-1]
        # Generate the position encoding matrix
        pos_encoding = torch.zeros((1, self.num_dim, k))
        position = torch.arange(0, self.num_dim, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float) * (-math.log(10000.0) / k))
        pos_encoding[:,:,0::2] = torch.sin(position * div_term)
        pos_encoding[:,:,1::2] = torch.cos(position * div_term)
        # Add the position encoding to the input tensor
        x = x + pos_encoding[:, :self.num_dim, :].to(self.device)
        return x

    def forward(self, x1, x2, mask_coef=None):
        # apply layer normalization
        x1, x2 = x1.view(-1, 1, self.input_size), x2.view(-1, self.num_dim, self.num_hi)
        x1 = x1.repeat(1, self.num_dim, 1)

        x = torch.concat((x1, x2), 2)
        x = self.add_position_encoding(x)
        # x1 = self.add_position_encoding(x1)
        # x2 = self.add_position_encoding(x2) 
        x = self.layer_norm(x)
        x = x.permute(0,2,1)
        # bs, n_seq, n_dim
        # print(x.size())
        # apply multi-head self-attention
        if mask_coef is not None:
            mask = torch.zeros([x.shape[1], x.shape[1]], dtype=torch.bool, device=self.device)
            mask[mask_coef, mask_coef] = True
            attn_output, _ = self.self_attn(x, x, x,attn_mask=mask)
        else:
            attn_output, _ = self.self_attn(x, x, x)

        x_attn = x + attn_output
        # apply feed-forward layers
        x = torch.flatten(x_attn, start_dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x, x_attn


class inferSpecFormer(nn.Module):
    def __init__(self, input_size, n_outputs, channels=32, hidden_size=128, dropout=0.2, device=torch.device('cpu')):
        super(inferSpecFormer, self).__init__()

        self.input_size = input_size
        self.device = device
        self.channels = channels
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size*channels, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 4*hidden_size, bias=True)
        self.fc3 = nn.Linear(4*hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, n_outputs, bias=True)
        self.do = nn.Dropout(p=dropout)

    def forward(self, x):
        # apply layer normalization
        # bs, n_seq, n_dim
        x = self.layer_norm(x.permute(0,2,1))
        x = self.do(x)
        x = x.reshape(x.shape[0], -1)
        # apply feed-forward layers
        # x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.fc1(x))
        # x = self.do(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class SpecFormer2(nn.Module):
    def __init__(self, input_size, n_outputs, tgt_mask=None, src_mask=None, n_hi=116, hidden_size=128, channels=64, num_heads=4, num_layers=4, dropout=0.1, device=torch.device('cpu')):
        super(SpecFormer2, self).__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.tgt_mask = tgt_mask
        self.src_mask = src_mask
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_dim = channels
        self.num_hi = n_hi
        self.device = device
        self.num_enc = input_size + n_hi

        self.layer_norm1 = nn.LayerNorm(channels)
        self.layer_norm2 = nn.LayerNorm(channels)

        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, 
        bias=False, dropout=dropout)

        self.fc1 = nn.Linear(self.num_dim, 4*self.num_dim, bias=False)
        self.fc2 = nn.Linear(4*self.num_dim, hidden_size,  bias=False)
        self.fc3 = nn.Linear(hidden_size, self.num_dim, bias=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout, dim_feedforward=4*channels, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc4 = nn.Linear(channels, 4*channels, bias=False)
        self.fc5 = nn.Linear(4*channels, channels, bias=False)
        self.fc6 = nn.Linear(channels*n_outputs, n_outputs, bias=False)

    def add_position_encoding(self, x):
        # Get the dimension of the data
        k = x.shape[-1]
        # Generate the position encoding matrix
        if k%2 != 0:
            k+=1 
        pos_encoding = torch.zeros((1, self.num_dim, k))
        position = torch.arange(0, self.num_dim, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float) * (-math.log(10000.0) / k))
        pos_encoding[:,:,0::2] = torch.sin(position * div_term)
        pos_encoding[:,:,1::2] = torch.cos(position * div_term)
        # Add the position encoding to the input tensor
        if k%2 != 0:
            x = x + pos_encoding[:, :self.num_dim, :-1].to(self.device)
        else:
            x = x + pos_encoding[:, :self.num_dim, :].to(self.device)
        return x.permute(0,2,1)

    def ffn(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.leaky_relu(self.fc3(x))
        return x

    def generate_tgt_mask(self, n_mask):
        tgt_mask = torch.triu(torch.ones(n_mask, n_mask), diagonal=1).bool().to(self.device)
        return tgt_mask

    def generate_src_mask(self, L, S, mask_coef):
        src_mask = torch.zeros([L, S], dtype=torch.bool, device=self.device)
        src_mask[mask_coef, mask_coef] = True
        return src_mask
    
    def generate_src_tgt_mask(self, L, S, mask_coef):
        src_mask = torch.zeros([L, S], dtype=torch.bool, device=self.device)
        src_mask[:, mask_coef] = True
        return src_mask

    def forward(self, x1, x2, tgt, mask_coef=None, infer=False):
        x1, x2 = x1.view(-1, 1, self.input_size), x2.view(-1, self.num_dim, self.num_hi)
        x1 = x1.repeat(1, self.num_dim, 1)
        x = torch.concat((x1, x2), 2)
        x = self.add_position_encoding(x)
        # bs, n_seq, n_dim

        if mask_coef is not None:
            src_mask = self.generate_src_mask(self.num_enc, self.num_enc, mask_coef)
            src_mask_tgt = self.generate_src_tgt_mask(self.input_size, self.num_enc, mask_coef)
        else:
            src_mask = None
            src_mask_tgt = None

        x = self.layer_norm1(x)
        attn_output, _ = self.self_attn(x, x, x,attn_mask=src_mask)

        src = x + attn_output
        # apply feed-forward layers
        src = self.ffn(src)
        """
        decoder

        """
        self.n_dec = tgt.shape[-1]
        tgt = self.add_position_encoding(tgt.view(-1,1,self.n_dec))
        tgt = self.layer_norm2(tgt)
        # bs, n_out, dim
        self.tgt_mask = self.generate_tgt_mask(self.n_dec)

        tgt = self.decoder(
            tgt=tgt, memory=src, 
            tgt_mask=self.tgt_mask,
            memory_mask=src_mask_tgt,
            )
        
        tgt = F.leaky_relu(self.fc4(tgt))
        tgt = self.fc5(tgt)
        tgt = torch.flatten(tgt, start_dim=1)
        tgt = self.fc6(tgt).view(-1, self.n_outputs)
        return tgt