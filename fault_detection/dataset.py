import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from metadata import METADATA
from utils import plot_distribution


class FaultDataset(Dataset):
    def __init__(self, dat_file="dataset/train_data.npz", normalize=True):
        self.seq_len = int(METADATA['time_step_detection'] / METADATA['time_step_control'])
        print(self.seq_len)
        # 读取 .dat 文件
        data = np.load(dat_file)
        self.input_data = data['inputs']
        self.target_data = data['targets']
        # plot_distribution(self.target_data)
        # print(self.input_data.shape, self.target_data.shape)

        print("-" * 25)
        print(np.max(self.input_data, axis=0), "\n", np.min(self.input_data, axis=0))
        print(np.max(self.target_data, axis=0), "\n", np.min(self.target_data, axis=0))

        max_omega = METADATA['max_omega']
        max_torque = METADATA['torque_range'][-1]
        min_eloss, max_eloss = METADATA['eloss_range']
        min_bias, max_bias = METADATA['bias_range']

        # input scale
        self.high_limit_input = np.array([0.75, 0.85, 0.85, max_torque, max_torque, max_torque])
        self.low_limit_input = - self.high_limit_input

        # target scale
        self.high_limit_output = np.array(
            [max_eloss, max_eloss, max_eloss, max_bias, max_bias, max_bias])
        self.low_limit_output = np.array(
            [min_eloss, min_eloss, min_eloss, min_bias, min_bias, min_bias])

        if normalize:
            self.input_data = (self.input_data - self.low_limit_input) / (self.high_limit_input - self.low_limit_input)
            print(np.max(self.input_data, axis=0), "\n", np.min(self.input_data, axis=0))

            # inertia_paras =  np.tile(METADATA['inertia_para'], (self.input_data.shape[0], 1))
            # self.input_data = np.hstack([self.input_data, inertia_paras])

            self.input_data = torch.tensor(self.input_data, dtype=torch.float32)

            self.target_data = (self.target_data - self.low_limit_output) / (
                    self.high_limit_output - self.low_limit_output)
            print(np.max(self.target_data, axis=0), "\n", np.min(self.target_data, axis=0))
            self.target_data = torch.tensor(self.target_data, dtype=torch.float32)
            print("=" * 25)

        else:
            self.input_data = torch.tensor(self.input_data, dtype=torch.float32)
            self.target_data = torch.tensor(self.target_data, dtype=torch.float32)

        # print(self.input_data.shape, self.target_data.shape)

    def __len__(self):
        return int(len(self.input_data) / self.seq_len)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx * self.seq_len: idx * self.seq_len + self.seq_len].view(self.seq_len, 6)
        target_sample = self.target_data[idx].view(1, 6)

        # input_sample = self.input_data[idx]
        # target_sample = self.target_data[idx]
        # print(input_sample.shape, target_sample.shape)

        return input_sample, target_sample


if __name__ == '__main__':
    # 加载数据集
    dataset = FaultDataset(dat_file="dataset/train_data.npz")

    # 打印数据集的长度（总样本数）
    print(f"Dataset length: {len(dataset)}")

    # for i in range(len(dataset)):
    #     input_sample, target_sample = dataset[i]
    # print(f"Input sample shape: {input_sample.shape}")
    # print(f"Target sample shape: {target_sample.shape}")
    # print(f"Input sample: {input_sample}")
    # print(f"Target sample: {target_sample}")
