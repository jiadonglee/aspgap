
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F


def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


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
        lr=1e-4,
    ):
        super().__init__()

        # self.save_hyperparameters()
        self.channels = channels
        self.n_outputs = n_outputs
        self.lr = lr
        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dropout=self.dropout,
            dim_feedforward=4*channels,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)
        self.input_projection = Linear(n_encoder_inputs, channels)
        # self.output_projection = Linear(n_decoder_inputs, channels)

        # self.linear = Linear(channels, 2)
        self.fc1 = Linear(channels, 64)
        self.fc2 = Linear(64, n_outputs)
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

    def forward(self, x):
        src = x
        
        src = self.encode_src(src) # (1, bs, 512)
        src = F.relu(src) # (1, bs, 512)
        
        src = src.permute(1, 0, 2) #(bs, 1, 512)
        src = src.view(-1, self.channels) # (bs, 512)

        src = self.fc1(src) # (bs, 64)
        src = F.relu(src)
        src = self.do(src)
        tgt = self.fc2(src) # (bs, 2)
        # out = self.decode_trg(trg=trg, memory=src)
        return tgt

    def training_step(self, batch, batch_idx):
        src, trg_out = batch['x'], batch['y']

        y_hat = self((src))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg_out = batch['x'], batch['y']

        y_hat = self((src))

        y_hat = y_hat.view(-1)
        y = trg_out.view(-1)

        loss = smape_loss(y_hat, y)
        self.log("valid_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        src, trg_out = batch['x'], batch['y']

        y_hat = self((src))

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


class Spec2HRd(nn.Module):
    def __init__(
        self,
        n_encoder_inputs,
        n_decoder_inputs,
        n_outputs,
        channels=512,
        dropout=0.2,
        lr=1e-4,
        nhead=4,
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
            nhead=nhead,
            dropout=self.dropout,
            dim_feedforward=4*channels,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=nhead,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.input_projection = Linear(n_encoder_inputs, channels)
        self.output_projection = Linear(n_decoder_inputs, channels)
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        # self.linear = Linear(channels, 2)
        self.fc1 = Linear(channels, 64)
        self.fc2 = Linear(64, n_outputs)
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

        return out

    def forward(self, x):
        src = x
        
        enc_ouput = self.encode_src(src) # (1, bs, 512)
        src = F.relu(enc_ouput) # (1, bs, 512)
        
        src = src.permute(1, 0, 2) #(bs, 1, 512)
        src = src.view(-1, self.channels) # (bs, 512)

        src = self.fc1(src) # (bs, 64)
        src = F.relu(src)
        
        tgt_a = self.fc2(src) # (bs, 2)

        dec_input = torch.concat((x.view(-1, self.n_encoder_inputs), tgt_a), dim=1).view(-1, 1, self.n_decoder_inputs)
        
        
        out = self.decode_trg(trg=dec_input, 
                              memory=enc_ouput)# (1, bs, 512)
        out = F.relu(out)
        out = out.permute(1, 0, 2) #(bs, 1, 512)
        out = out.view(-1, self.channels) # (bs, 512)
        
        out = self.fc1(out) # (bs, 64)
        out = F.relu(out)
        
        out = self.fc2(out) # (bs, 2)
        return torch.concat((tgt_a, out), dim=1)