import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(x), dim=1)  # [B, seq_len, 1]
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # [B, hidden_size]
        # 使用concat的方式进行残差连接，直接使用最后一个时间步的值
        residual = x[:, -1, :]  # 取最后一个时间步的值作为残差 [B, hidden_size]
        concat_output = torch.cat([weighted_sum, residual], dim=1)  # [B, hidden_size * 2]
        return concat_output

class FaultNet(nn.Module):
    def __init__(self, configs, input_size=6, hidden_size=128, output_size=6, num_layers=1, bidirectional=True, dropout=0.2):
        super(FaultNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)
        # 因为concat后维度翻倍，所以fc的输入维度需要调整
        self.fc = nn.Linear(hidden_size * 4 if bidirectional else hidden_size * 2, output_size)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_size * 2]
        attn_out = self.attention(lstm_out)  # [B, hidden_size * 2]
        output = self.fc(attn_out)  # [B, output_size]

        e_output = torch.sigmoid(output[:, :3])
        ua_output = output[:, 3:]
        final_output = torch.cat([e_output, ua_output], dim=1)

        return final_output.unsqueeze(1)  # [batch, 1, 6]