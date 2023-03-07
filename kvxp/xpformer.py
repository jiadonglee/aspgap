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
- xpformer model script
"""
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class XPformer(nn.Module):
    def __init__(self, input_size, n_outputs, hidden_size=16, channels=10, num_heads=2, num_layers=2, dropout=0.2, device=torch.device('cpu')):

        super(XPformer, self).__init__()
        self.input_size = input_size
        self.num_dim = channels
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout, dim_feedforward=4*channels, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels, nhead=num_heads, dropout=dropout, dim_feedforward=4*channels, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_project = nn.Linear(1, self.num_dim)
        self.fc1 = nn.Linear(channels, hidden_size,  bias=True)
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def add_position_encoding(self, x):
        # Get the dimension of the data
        k = 12
        # Generate the position encoding matrix
        pos_encoding = torch.zeros((1, self.num_dim, k))
        position = torch.arange(0, self.num_dim, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float) * (-math.log(10000.0) / k))
        pos_encoding[:,:,0::2] = torch.sin(position * div_term)
        pos_encoding[:,:,1::2] = torch.cos(position * div_term)
        # Add the position encoding to the input tensor
        x = x + pos_encoding[:, :self.num_dim, :-1].to(self.device)
        return x.permute(0,2,1)

    def forward(self, x=None, y=None, tgt_mask=None):
        x = x.reshape(-1, self.num_dim, self.input_size)
        x = self.add_position_encoding(x)

        src = self.encoder(x) #(bs, 11, 10)
        tgt = self.out_project(y.view(-1, self.n_outputs, 1))
        tgt = self.decoder(tgt, memory=src, tgt_mask=tgt_mask) #(bs, 4, n_dim)
        # tgt = torch.flatten(tgt, start_dim=1)
        tgt = F.leaky_relu(self.fc1(tgt))
        tgt = self.fc2(tgt)
        return tgt.view(-1, self.n_outputs)
       

class XPformer2(nn.Module):
    def __init__(self, input_size, n_outputs, input_proj=True, output_proj=False, 
                 hidden_size=16, channels=8, num_heads=2, num_layers=2,
                 dropout=0.1, device=torch.device('cpu')):

        super(XPformer2, self).__init__()
        self.input_size = input_size
        self.num_dim = channels
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.input_proj = input_proj
        self.output_proj = output_proj

        self.drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dropout=dropout, dim_feedforward=4*channels, 
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_project = nn.Linear(1, self.num_dim)
        # self.out_project = nn.Linear(1, self.num_dim)
        self.fc1 = nn.Linear(channels*input_size, hidden_size,  bias=True)
        self.fc2 = nn.Linear(hidden_size, n_outputs, bias=True)

    def add_position_encoding(self, x):
        # Get the dimension of the data
        k = self.input_size
        # Generate the position encoding matrix
        pos_encoding = torch.zeros((1, self.num_dim, k))
        position = torch.arange(0, self.num_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float) * (-math.log(10000.0) / k))
        pos_encoding[:,:,0::2] = torch.sin(position * div_term)
        pos_encoding[:,:,1::2] = torch.cos(position * div_term)

        # Add the position encoding to the input tensor
        x = x + pos_encoding[:, :self.num_dim, :].to(self.device)
        return x.permute(0,2,1)

    def forward(self, x=None, src_mask=None):
        # x = x.reshape(-1, self.num_dim, self.input_size)
        if self.input_proj:
            x = x.reshape(-1, self.input_size, 1)
            x = self.input_project(x)
        else:
            x = x.reshape(-1, self.input_size, self.num_dim)

        x = self.add_position_encoding(x.view(-1, self.num_dim, self.input_size))
        x = self.drop(x)
        tgt = self.encoder(x, src_key_padding_mask=src_mask) #(bs, 11, 10)
        tgt = self.drop(tgt)
        tgt = torch.flatten(tgt, start_dim=1)
        tgt = F.leaky_relu(self.fc1(tgt))
        tgt = self.fc2(tgt)
        return tgt.view(-1, self.n_outputs)
    


class XPformerConv(nn.Module):
    def __init__(self, input_size, n_outputs, input_proj=True, output_proj=False, 
                 hidden_size=16, channels=8, num_heads=2, num_layers=2,
                 dropout=0.1, device=torch.device('cpu')):
        super(XPformerConv, self).__init__()
        self.input_size = input_size
        self.num_dim = channels
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.input_proj = input_proj
        self.output_proj = output_proj

        self.drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dropout=dropout, dim_feedforward=4*channels, 
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_project = nn.Linear(1, self.num_dim)
        # self.out_project = nn.Linear(1, self.num_dim)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(channels, 2*channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(channels, 4*channels, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4*channels*input_size, hidden_size,  bias=True)
        self.fc2 = nn.Linear(hidden_size, n_outputs, bias=True)

    def add_position_encoding(self, x):
        # Get the dimension of the data
        k = self.input_size
        # Generate the position encoding matrix
        pos_encoding = torch.zeros((1, self.num_dim, k))
        position = torch.arange(0, self.num_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float) * (-math.log(10000.0) / k))
        pos_encoding[:,:,0::2] = torch.sin(position * div_term)
        pos_encoding[:,:,1::2] = torch.cos(position * div_term)

        # Add the position encoding to the input tensor
        x = x + pos_encoding[:, :self.num_dim, :].to(self.device)
        return x.permute(0,2,1)

    def forward(self, x=None, src_mask=None):
        # x = x.reshape(-1, self.num_dim, self.input_size)
        if self.input_proj:
            x = x.reshape(-1, self.input_size, 1)
            x = self.input_project(x)
        else:
            x = x.reshape(-1, self.input_size, self.num_dim)

        x = self.add_position_encoding(x.view(-1, self.num_dim, self.input_size))
        x = self.drop(x)
        tgt = self.encoder(x, src_key_padding_mask=src_mask) #(bs, 11, 10)
        tgt = self.drop(tgt)
        # tgt = torch.flatten(tgt, start_dim=1)
        # tgt = F.leaky_relu(self.fc1(tgt))
        # tgt = self.fc2(tgt)
        tgt = tgt.permute(0,2,1)
        tgt = self.conv1(tgt)
        tgt = self.conv2(tgt)
        tgt = torch.flatten(tgt, start_dim=1)
        tgt = F.leaky_relu(self.fc1(tgt))
        tgt = self.fc2(tgt)
        return tgt
       
class CNN(nn.Module):
    def __init__(self, n_input=7514, n_output=4, in_channel=1):
        super().__init__()
        self.n_input = n_input
        self.in_channel=in_channel
        self.layer_norm = nn.LayerNorm(in_channel)
        self.conv1 = nn.Conv1d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * n_input, 128)
        self.fc2 = nn.Linear(128, n_output)
        
        
    def forward(self, x):
        # x = self.layer_norm(x)
        x = x.reshape(-1, self.in_channel, self.n_input)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.2):
        super(MLP, self).__init__()
        self.input_size = input_size
        # self.layer_norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, 2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 128)
        self.fc5 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x



class MLP_upsampling(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.2):
        super(MLP_upsampling, self).__init__()
        self.input_size = input_size
        # self.layer_norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, 4*hidden_size)
        self.fc4 = nn.Linear(4*hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
