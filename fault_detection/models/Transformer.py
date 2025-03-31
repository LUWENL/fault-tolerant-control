import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, configs, input_size = 6, hidden_size = 128, output_size = 6, num_heads = 4, num_layers = 1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 50, hidden_size))  # 假设序列长度为50
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, x)  # 自注意力
        x = x.mean(dim=1)  # 全局平均池化
        return self.fc(x).unsqueeze(1)