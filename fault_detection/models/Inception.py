import torch
import torch.nn as nn
import math

class InceptionModule(nn.Module):
    def __init__(self, in_channels=6, out_channels=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, 32, kernel_size=1) if in_channels > 1 else nn.Identity()

        self.convs = nn.ModuleList([
            nn.Conv1d(32, out_channels, kernel_size=10, padding='same'),
            nn.Conv1d(32, out_channels, kernel_size=20, padding='same'),
            nn.Conv1d(32, out_channels, kernel_size=40, padding='same'),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        ])
        self.conv_out = nn.Conv1d(out_channels * 3 + 32, out_channels, kernel_size=1) if in_channels > 1 else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)
        features = [conv(x) for conv in self.convs[:3]]
        features.append(self.convs[3](x))
        out = torch.cat(features, dim=1)
        if hasattr(self, 'conv_out'):
            out = self.conv_out(out)
        return out


class InceptionTime(nn.Module):
    def __init__(self, configs, input_dim=6, num_classes=6):
        super().__init__()
        self.inception1 = InceptionModule(input_dim, 32)
        self.inception2 = InceptionModule(32, 64)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch,9,5]
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avg_pool(x).squeeze(-1)
        out = self.fc(x)
        e_output = torch.sigmoid(out[:, :3])
        ua_output = out[:, 3:]
        return torch.cat([e_output, ua_output], dim=1).unsqueeze(1)