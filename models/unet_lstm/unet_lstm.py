import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x_seq):  # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.size()
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x_seq.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        return h  # output last hidden state

class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, 1)
        )
    def forward(self, x):
        return self.net(x)

class UNetConvLSTM(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, feat_dim=64, lstm_hidden=128):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, feat_dim)
        self.temporal = ConvLSTM(input_dim=feat_dim*2, hidden_dim=lstm_hidden)
        self.decoder = CNNDecoder(lstm_hidden, out_channels)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, H, W = x.size()
        x = x.view(B*T, 1, H, W)
        feat = self.encoder(x)  # (B*T, F, H, W)
        feat = feat.view(B, T, feat.size(1), H, W)
        h = self.temporal(feat)
        out = self.decoder(h)
        return out