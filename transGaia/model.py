
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device))==1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Spec2label(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_outputs,
        channels=64,
        n_heads=8,
        n_layers=8,
        dropout=0.2,
        attn=False,
        lr=1e-4,
    ):
        super().__init__()

        # self.save_hyperparameters()
        self.channels = channels
        self.n_outputs = n_outputs
        self.lr = lr
        self.dropout = dropout
        self.input_pos_embedding  = torch.nn.Embedding(128, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(128, embedding_dim=channels)
        self.attn = attn

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dropout=self.dropout,
            dim_feedforward=4*channels,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)
        self.input_projection = nn.Linear(n_encoder_inputs, channels)
        # self.output_projection = Linear(n_decoder_inputs, channels)
        # self.linear = Linear(channels, 2)
        self.fc1 = nn.Linear(channels, 64)
        self.fc2 = nn.Linear(64, n_outputs)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)

        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
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
        src = x
        if self.attn:
            src, attention_maps = self.encode_src(src) # (1, bs, 512)
        else:
            src = self.encode_src(src)
            
        src = F.relu(src) # (1, bs, 512)
        src = src.permute(1, 0, 2) #(bs, 1, 512)
        src = src.view(-1, self.channels) # (bs, 512)

        src = self.fc1(src) # (bs, 64)
        src = F.relu(src)
        src = self.do(src)
        tgt = self.fc2(src) # (bs, 2)

        if self.attn:
            return tgt, attention_maps
        else:
            return tgt



class Spec2HRd(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        n_outputs,
        channels=512,
        dropout=0.2,
        lr=1e-4,
        n_heads=4,
        n_layers=8,
        mode="train"
    ):
        super().__init__()
        self.n_encoder_inputs = n_encoder_inputs
        self.n_decoder_inputs = n_decoder_inputs
        # self.save_hyperparameters()
        self.channels = channels
        self.n_outputs = n_outputs
        self.lr = lr
        self.dropout = dropout
        self.mode = mode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dropout=self.dropout,
            dim_feedforward=4*channels,
        )
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=channels,
        #     nhead=n_heads,
        #     dropout=self.dropout,
        #     dim_feedforward=4*channels,
        # )
        self.channels = channels

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection_a = Linear(n_decoder_inputs, channels)
        self.output_projection_b = Linear(n_decoder_inputs+2, channels)
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        # self.linear = Linear(channels, 2)
        self.fc1 = Linear(channels, 64)
        self.fc2 = Linear(64, n_outputs)
        self.fc3 = Linear(64, 1)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)
        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_src(self, src):
        if src.size()[2] == self.n_decoder_inputs:
            src_start = self.output_projection_a(src)
        else:
            src_start = self.output_projection_b(src)

        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def forward(self, x):
        src = self.encode_src(x).squeeze(1) # (bs, channels)
        src = self.fc1(src) # (bs, 64)
        tgt_a = self.fc2(F.relu(src)) # (bs, 2) (teff, logg)

        dec_input_a = torch.concat((x.view(-1, self.n_encoder_inputs), tgt_a), dim=1)
        dec_input_a = dec_input_a.view(-1, 1, self.n_decoder_inputs) #(bs, 30+2)

        out = self.decode_src(dec_input_a).squeeze(1) # (bs, channels)
        out = self.fc1(out) # (bs, 64)
        tgt_b = self.fc2(F.relu(out)) # (bs, 2) (parallax, [M/H]])

        dec_input_b = torch.concat((x.view(-1, self.n_encoder_inputs), tgt_a, tgt_b), dim=1).view(-1, 1, self.n_decoder_inputs+2) #(bs, 30+4)

        out = self.decode_src(dec_input_b).squeeze(1) # (bs, channels)
        tgt_c = self.fc1(out) # (bs, 64)
        tgt_c = self.fc3(F.relu(tgt_c)) # (bs, 1)
        
        if self.mode=='train':
            return torch.concat((tgt_a, tgt_b, tgt_c), dim=1).view(-1, 5)
        
        else:
            return out

    
    
    
    
class Spec2HRd_err(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        n_outputs,
        channels=512,
        dropout=0.2,
        lr=1e-4,
        n_heads=4,
        n_layers=8,
    ):
        super().__init__()
        self.n_encoder_inputs = n_encoder_inputs
        self.n_decoder_inputs = n_decoder_inputs
        # self.save_hyperparameters()
        self.channels = channels
        self.n_outputs = n_outputs
        self.lr = lr
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dropout=self.dropout,
            dim_feedforward=4*channels,
        )
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=channels,
        #     nhead=n_heads,
        #     dropout=self.dropout,
        #     dim_feedforward=4*channels,
        # )
        self.channels = channels

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection_a = Linear(n_decoder_inputs, channels)
        self.output_projection_b = Linear(n_decoder_inputs+2, channels)
        self.output_projection_err = Linear(n_decoder_inputs+3, channels)

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        # self.linear = Linear(channels, 2)
        self.fc1 = Linear(channels, 64)
        self.fc2 = Linear(64, n_outputs)
        self.fc3 = Linear(64, 1)
        self.fc_err = Linear(64, 5)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src):
        src_start = self.input_projection(src)
        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_src(self, src):
        if src.size()[2] == self.n_decoder_inputs:
            src_start = self.output_projection_a(src)

        elif src.size()[2] == self.n_decoder_inputs+2:
            src_start = self.output_projection_b(src)

        else:
            src_start = self.output_projection_err(src)

        in_sequence_len, batch_size = src_start.size(1), src_start.size(0)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def forward(self, x):
        src = self.encode_src(x).squeeze(1) # (bs, channels)
        src = self.fc1(src) # (bs, 64)
        tgt_a = self.fc2(F.relu(src)) # (bs, 2) (teff, logg)

        dec_input_a = torch.concat((x.view(-1, self.n_encoder_inputs), tgt_a), dim=1)
        dec_input_a = dec_input_a.view(-1, 1, self.n_decoder_inputs) #(bs, 30+2)

        out = self.decode_src(dec_input_a).squeeze(1) # (bs, channels)
        out = self.fc1(out) # (bs, 64)
        tgt_b = self.fc2(F.relu(out)) # (bs, 2) (parallax, [M/H]])

        dec_input_b = torch.concat((x.view(-1, self.n_encoder_inputs), tgt_a, tgt_b), dim=1).view(-1, 1, self.n_decoder_inputs+2) #(bs, 30+4)

        out = self.decode_src(dec_input_b).squeeze(1) # (bs, channels)
        out = self.fc1(out) # (bs, 64)
        tgt_c = self.fc3(F.relu(out)) # (bs, 1)
        output = torch.concat((tgt_a, tgt_b, tgt_c), dim=1).view(-1, 5)

        dec_input_err = torch.concat((x.view(-1, self.n_encoder_inputs), output), dim=1).view(-1, 1, self.n_decoder_inputs+3)
        e_out = self.decode_src(dec_input_err).squeeze(1)
        e_out = self.fc1(e_out) # (bs, 64)
        e_out = self.fc_err(F.relu(e_out)) # (bs, 2) (parallax, [M/H]])

        return torch.concat((output, e_out), dim=1).view(-1, 10)