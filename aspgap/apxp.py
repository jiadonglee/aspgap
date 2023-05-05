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





class CNN2(nn.Module):
    def __init__(self, n_input=128, n_output=4, in_channel=1, n_dim=64, hidden_size=128):
        super().__init__()
        self.n_input = n_input
        self.in_channel=in_channel
        self.n_dim = n_dim
        self.layer_norm = nn.LayerNorm(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * n_input * n_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_output)
        
        
    def forward(self, x):
        # x = self.layer_norm(x)
        x = x.reshape(-1, 1, self.n_input, self.n_dim)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x