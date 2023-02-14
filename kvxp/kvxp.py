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


class xp2label_nn(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, channels=128, dropout=0.2):
        super().__init__()

        # self.save_hyperparameters()
        self.n_encoder_inputs = n_encoder_inputs
        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.input_pos_embedding = torch.nn.Embedding(
            1024, embedding_dim=channels
        )

        self.feedfoward1 = nn.Linear(channels, 4*channels)
        self.feedfoward2 = nn.Linear(4*channels, channels)
        self.input_projection = nn.Linear(1, channels)

        self.fc1 = nn.Linear(channels*n_encoder_inputs, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_outputs)
        self.do  = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)
        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0).repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)
        src = src_start + pos_encoder
        return src


    def forward(self, x):
        src = x.view(-1, self.n_encoder_inputs, 1) # (bs, 1, n_enc)
        src = self.encode_src(src)
        src = F.leaky_relu(self.feedfoward1(src)) # (bs, n_enc, 4*dim)
        src = F.leaky_relu(self.feedfoward2(src))  # (bs, n_enc, dim)
        
        src = torch.flatten(src, start_dim=1) #(bs, 512*113)
        src = F.leaky_relu(self.fc1(src)) # (bs, 1024)
        src = F.leaky_relu(self.fc2(src)) # (bs, 128)
        src = F.leaky_relu(self.fc3(src)) # (bs, 1)
        tgt = self.do(src)
        return tgt


class xp2label_attn(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, n_heads=8,
                n_layers=8, channels=128, dropout=0.2, coef_mask=None, 
                device=torch.device('cuda:0')):
        super().__init__()

        # self.save_hyperparameters()
        self.n_encoder_inputs = n_encoder_inputs
        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.device = device
        self.coef_mask = coef_mask

        self.enc_embedding = DataEmbedding(c_in=1, d_model=channels, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dropout=self.dropout, dim_feedforward=4*channels, 
            batch_first=True, 
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(n_encoder_inputs, channels)

        self.fc1 = nn.Linear(channels*n_encoder_inputs, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, n_outputs)
        self.do  = nn.Dropout(p=self.dropout)


    def forward(self, x):
        src = self.enc_embedding(x.view(-1, self.n_encoder_inputs, 1)) 
        # (bs, n_enc, dim)

        if self.coef_mask is not None:
            self.mask = torch.ones(x.size(0), self.n_encoder_inputs).to(self.device)
            self.mask[:, self.coef_mask] = 0
            self.mask = self.mask.to(torch.bool)
        else:
            self.mask = None

        src = self.encoder(src, src_key_padding_mask=self.mask) # (bs, n_enc, dim)
        
        src = torch.flatten(src, start_dim=1) #(bs, 512*113)
        src = F.relu(self.fc1(src)) # (bs, 1024)
        src = F.relu(self.fc2(src)) # (bs, 128)
        src = F.relu(self.fc3(src)) # (bs, 1)
        tgt = self.do(src)

        return tgt


class xp2label(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, channels=64, n_heads=8, n_layers=8, dropout=0.2, attn=False, coef_mask=None, device=torch.device('cpu')):
        super().__init__()

        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.input_pos_embedding = torch.nn.Embedding(10000, embedding_dim=channels)
        self.attn = attn
        self.device = device
        self.coef_mask = coef_mask
        self.n_enc = n_encoder_inputs

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dropout=self.dropout, dim_feedforward=4*channels, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(1, channels)
        # self.output_projection = Linear(n_decoder_inputs, channels)
        self.fc0 = nn.Linear(n_encoder_inputs, 256)
        self.fc1 = nn.Linear(channels*256, 128)
        self.fc2 = nn.Linear(128, 64)
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

        if self.coef_mask is not None:
            self.mask = torch.zeros_like(src[:,:,0], dtype=torch.bool, device=self.device)
            self.mask[:, self.coef_mask] = 1
        else:
            self.mask = None

        src = self.encoder(src, src_key_padding_mask=self.mask) # (bs, n_enc, dim)

        if self.attn:
            attention_maps = []
            for l in self.encoder.layers:
                attention_maps.append(l.self_attn(src, src, src)[1])
            return src, attention_maps
        else:
            return src

    def forward(self, x):
        src = x.view(-1, 1, self.n_enc)
        src = F.leaky_relu(self.fc0(src)).permute(0,2, 1)
        if self.attn:
            src, attention_maps = self.encode_src(src) # (1, bs, 512)
        else:
            src = self.encode_src(src)
        src_enc = src
    
        src = torch.flatten(src, start_dim=1)
        src = self.do(src)
        src = F.leaky_relu(self.fc1(src))
        src = F.leaky_relu(self.fc2(src))
        tgt = F.leaky_relu(self.fc3(src)) # (bs, 1)
        
        if self.attn:
            return tgt, attention_maps, src_enc
        else:
            return tgt


class xp2label_seg(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, n_hi=7514, channels=64, n_heads=8, n_layers=8, dropout=0.2, attn=False):
        super().__init__()

        self.channels = channels
        self.n_outputs = n_outputs
        self.dropout = dropout
        self.input_pos_embedding = torch.nn.Embedding(256, embedding_dim=channels)
        self.attn = attn
        self.n_enc = n_encoder_inputs
        self.n_hi  = n_hi
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dropout=self.dropout, 
            dim_feedforward=4*channels, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_projection = nn.Linear(1, channels)

        # self.output_projection = Linear(n_decoder_inputs, channels)
        self.downsample = nn.Linear(self.n_hi, self.n_enc)
        self.fc1 = nn.Linear(channels*self.n_enc*2, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, n_outputs)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)
        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)
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
        src = x.view(-1, self.n_enc+self.n_hi, 1)
        
        src_a = src[:,-self.n_hi:,:].view(-1, 1, self.n_hi)
        
        src_a = self.downsample(src_a).view(-1, self.n_enc, 1) 
        
        src = torch.concat(
            (src[:, :self.n_enc, :], src_a), 1
         )
        
        if self.attn:
            src, attention_maps = self.encode_src(src) # (1, bs, 512)
        else:
            src = self.encode_src(src)
            
        src_enc = src
    
        src = torch.flatten(src, start_dim=1)

        src = self.do(src)
        src = F.leaky_relu(self.fc1(src))
        src = F.leaky_relu(self.fc2(src))
        tgt = F.leaky_relu(self.fc3(src)) # (bs, 1)
        
        if self.attn:
            return tgt, attention_maps, src_enc
        else:
            return tgt

class specformer(nn.Module):
    def __init__(self, n_encoder_inputs, n_outputs, 
                 n_hi=7514, channels=32, n_heads=8, 
                 n_layers=8, n_token=512, dropout=0.2, attn=False,
                 coef_mask=None, device=torch.device('cpu')):
        super().__init__()

        self.channels = channels
        self.n_outputs = n_outputs
        # self.input_pos_embedding = torch.nn.Embedding(256, embedding_dim=channels)
        # self.input_projection = nn.Linear(1, channels)
        self.dropout = dropout
        self.attn = attn
        self.n_enc = n_encoder_inputs
        self.n_hi  = n_hi
        self.coef_mask = coef_mask
        self.device = device
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads,
            dropout=self.dropout, 
            dim_feedforward=4*channels, 
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.conv0 = nn.Conv1d(1, channels, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, stride=3, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=3, padding=0)

        self.fc1 = nn.Linear(channels*386, 128)
        self.fc2 = nn.Linear(128, n_outputs)
        self.do = nn.Dropout(p=self.dropout)
        
        
    def encode_src(self, src):
        
        if self.attn:
            attention_maps = []
            for l in self.encoder.layers:
                attention_maps.append(l.self_attn(src, src, src)[1])
        else:
            attention_maps = None
            
        if self.coef_mask is not None:
            self.mask = torch.zeros_like(src[:,:,0], dtype=torch.bool)
            self.mask[:, self.coef_mask] = True
        else:
            self.mask = None
        
         # (bs, n_enc, dim)
        src = self.encoder(src, src_key_padding_mask=self.mask)
        return src


    def forward(self, x):
        src = x.view(-1, 1, self.n_enc+self.n_hi)
        src_a = src[:, :, -self.n_hi:]
        src_a = F.leaky_relu(self.conv1(src_a))
        src_a = F.leaky_relu(self.conv2(src_a))
        src_a = F.leaky_relu(self.conv2(src_a)) # (bs, dim, 278)
        
        src_x = F.leaky_relu(self.conv0(src[:, :, :self.n_enc])) #(bs, dim, 108)
        
        src = torch.concat(
            (src_x, src_a), 2
         ).permute(0,2,1) # (bs, 110+278, dim)
        
        if self.attn:
            src, attention_maps = self.encode_src(src)
        else:
            src = self.encode_src(src)
            
        src_enc = src
        src = torch.flatten(src, start_dim=1)
        src = self.do(src)
        
        src = F.leaky_relu(self.fc1(src))
        tgt = F.leaky_relu(self.fc2(src))
        
        if self.attn:
            return tgt, attention_maps, src_enc
        else:
            return tgt
        
        
class CNN(nn.Module):
    def __init__(self, n_input=7514, n_output=4):
        super().__init__()
        self.n_input = n_input
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * n_input, 128)
        self.fc2 = nn.Linear(128, n_output)
        
    def forward(self, x):
        x = x.view(-1, 1, self.n_input)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x