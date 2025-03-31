import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, configs, hidden_sizes = [64, 128, 256]):
        super(MLP, self).__init__()

        self.input_size = configs.label_len * configs.seq_len
        self.output_size = configs.num_class
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return x.unsqueeze(1)
