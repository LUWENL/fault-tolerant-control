import random
from operator import index

import torch
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from numpy import cos, sin
from metadata import METADATA
from tolerant_control.fault_tolerant_gym.models.dynamics_and_kinematics import integrate_omega, \
    generate_external_disturbance


def get_weight(el_para):
    '''
    el_para: effective loss, shape = [N_actuator]

    weight: shape = [N_actuator, N_actuator]
    '''

    return torch.diag(torch.ones_like(el_para) - el_para)


def from_weight_to_eloss(weight):
    eloss0 = 1 - weight[0][0]
    eloss1 = 1 - weight[1][1]
    eloss2 = 1 - weight[2][2]
    return np.array([eloss0, eloss1, eloss2])


def fit_weight_and_u_a(part_left, u_c):
    """
    part_left: (N, 3) 矩阵，每行为一个 part_left^T
    u_c: (N, 3) 矩阵，每行为一个 u_c^T
    返回：
        weight: (3, 3) 矩阵
        u_a: (3, 1) 列向量
    """
    N = part_left.shape[0]  # 数据组数
    X = np.hstack((u_c, np.ones((N, 1))))  # 扩展 U_c，增加一列 1
    Y = part_left
    for i in range(Y.shape[0]):
        d = generate_external_disturbance(t=random.randint(0, 300),
                                          omega=np.random.uniform(-METADATA['max_omega'], METADATA['max_omega'],
                                                                  size=3))
        Y[i] += d

    # 计算最小二乘解
    Theta = np.linalg.lstsq(X, Y, rcond=None)[0]  # 计算最小二乘

    weight = Theta[:3, :].T  # 提取 W^T 再转置
    eloss = from_weight_to_eloss(weight)
    u_a = Theta[3, :].reshape(3)  # 提取 u_a^T 并转置为列向量

    return eloss, u_a


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    J = METADATA['inertia_matrix']

    data = np.load("dataset/test_data.npz")
    input_data = data['inputs']
    target_data = data['targets']
    print(target_data.shape)
    seq_len = int(METADATA['time_step_detection'] / METADATA['time_step_control'])

    # 转换为 PyTorch 张量并移动到 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    all_MSE = []
    all_MAE = []

    start_time = time.time()

    for idx in range(int(len(input_data) / seq_len)):
        # for idx in range(int(60000)):
        part_left = input_data[idx * seq_len: idx * seq_len + seq_len].reshape(seq_len, 6)[:, :3]

        u_c = input_data[idx * seq_len: idx * seq_len + seq_len].reshape(seq_len, 6)[:, 3:]
        el_true = target_data[idx].reshape(6)[:3]
        ua_true = target_data[idx].reshape(6)[3:]
        fault_info = target_data[idx].reshape(6)

        print("Index: {}".format(idx))
        print("Truth Fault: {}, {}".format(el_true, ua_true))

        estimated_el, estimated_ua = fit_weight_and_u_a(part_left, u_c)
        print(estimated_el.shape, estimated_ua.shape)
        estimated_fault_info = np.concatenate([estimated_el, estimated_ua])
        print("Estimated Fault: {}".format(estimated_fault_info))

        MSE = np.mean((fault_info - estimated_fault_info) ** 2)
        MAE = np.mean(np.abs(fault_info - estimated_fault_info))
        print("MSE : {} , MAE: {}".format(MSE, MAE))
        print("-" * 25)

        all_MSE.append(MSE)
        all_MAE.append(MAE)

    end_time = time.time()
    mean_inference_time = (end_time - start_time) / len(all_MSE)

    print(np.mean(all_MSE), np.mean(all_MAE), mean_inference_time)
