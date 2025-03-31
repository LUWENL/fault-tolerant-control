import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, configs, hidden_size=128):
        """
        LSTM 模型
        """
        input_size = 6
        output_size = 6

        if configs != "default":
            input_size = configs.label_len
            output_size = configs.num_class

        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)


        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # LSTM层输出包含所有时间步的结果
        lstm_out, _ = self.lstm(x)  # lstm_out形状: (batch, seq_len=50, hidden_size)

        # 只取最后一个时间步的输出
        last_time_step = lstm_out[:, -1, :]  # 形状: (batch, hidden_size)

        # 全连接层生成最终输出
        output = self.fc(last_time_step)  # 形状: (batch, 6)
        return output.unsqueeze(1)



if __name__ == '__main__':
    batch_size = 8
    input_data = torch.randn(batch_size, 5, 9)

    # 假设 target 数据是随机的 [batch_size, 8]，用于后期对比输出
    target_data = torch.randn(batch_size, 3)

    # 初始化 MLP 模型
    input_size = 9  # 每个输入样本有 5 个时间步和 17 个特征
    hidden_sizes = 128  # 隐藏层的神经元数量
    output_size = 6  # 输出层的神经元数量

    model = LSTMModel(input_size, hidden_sizes)

    # 前向传播：传入输入数据
    output = model(input_data)  # 需要将输入数据展平为 [batch_size, 90]

    # 打印输出结果
    print("\nInput Data:")
    print(input_data.shape)
    print("\nOutput Data:")
    print(output.shape)
