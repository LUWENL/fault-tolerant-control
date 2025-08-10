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


def calculate_error(el_para_hat, ua_para_hat, u_c, part_left, sim_duration=METADATA['time_step_detection'],
                    dt=METADATA['time_step_control']):
    error = torch.zeros(1, device=device)
    for i in range(int(sim_duration / dt)):
        weight = get_weight(el_para_hat) 
        u_actual = (weight @ u_c[i]) + ua_para_hat  
        d = generate_external_disturbance(t = random.randint(0, 300), omega = np.random.uniform(-METADATA['max_omega'], METADATA['max_omega'], size = 3))
        d = torch.from_numpy(d).to(device)

        error += torch.norm(part_left[i] - u_actual - d)

    return error  


def lr_lambda(step, threshold=80, gamma=0.992):
    if step <= threshold:
        return gamma ** step  # 前30步使用 gamma=0.995 的 StepLR
    else:
        # return (gamma ** threshold) * (0.95 ** ((step - threshold) // 5))
        return (gamma - 0.005) ** step  # 前30步使用 gamma=0.995 的 StepLR


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    J = METADATA['inertia_matrix']

    data = np.load("dataset/test_data.npz")
    input_data = data['inputs']
    target_data = data['targets']
    print(target_data.shape)
    seq_len = int(METADATA['time_step_detection'] / METADATA['time_step_control'])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    Iterations_num = 300
    lr = 1e-1 

    all_MSE = []
    all_MAE = []

    start_time = time.time()

    for idx in range(int(len(input_data) / seq_len)):
    # for idx in range(360):
        part_left = torch.from_numpy(input_data[idx * seq_len: idx * seq_len + seq_len].reshape(seq_len, 6)[:, :3]).to(
            torch.float32).to(device)
        u_c = torch.from_numpy(input_data[idx * seq_len: idx * seq_len + seq_len].reshape(seq_len, 6)[:, 3:]).to(
            torch.float32).to(device)
        el_true = target_data[idx].reshape(6)[:3]
        ua_true = target_data[idx].reshape(6)[3:]
        fault_info = target_data[idx].reshape(6)

        print("Index: {}".format(idx))
        print("Truth Fault: {}, {}".format(el_true, ua_true))

        # for estimate
        el_para = torch.zeros(METADATA['N_actuator'], device=device)
        ua_para = torch.zeros(METADATA['N_actuator'], device=device)

        el_para.requires_grad_(True)
        ua_para.requires_grad_(True)

        optimizer = torch.optim.Adam([el_para, ua_para], lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)  # 学习率调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)  # 使用LambdaLR

        for iteration in range(Iterations_num):
            optimizer.zero_grad()  
            error = calculate_error(el_para, ua_para, u_c, part_left)
            error.sum().backward()  
            optimizer.step()
            scheduler.step()

            # print(el_para.detach().clone(), ua_para.detach().clone())

        estimated_el = el_para.detach().clone().cpu().numpy()
        estimated_ua = ua_para.detach().clone().cpu().numpy()
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

