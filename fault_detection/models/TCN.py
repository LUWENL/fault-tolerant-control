import torch
import torch.nn as nn


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class TCN(nn.Module):
    def __init__(self, configs, num_channels=[32, 64, 128, 256]):
        super().__init__()
        layers = []
        in_ch = 6
        output_dim = 6

        if configs != "default":
            in_ch = configs.label_len
            output_dim = configs.num_class

        for out_ch in num_channels:
            layers += [DilatedBlock(in_ch, out_ch, dilation=2 ** len(layers))]
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1) # x: [batch,50,6] -> [batch,6,50]
        features = self.blocks(x)
        pooled = torch.mean(features, dim=2)
        out = self.fc(pooled)

        return torch.sigmoid(out).unsqueeze(1)
