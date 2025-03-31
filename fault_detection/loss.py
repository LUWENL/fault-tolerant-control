import numpy as np
import torch
from metadata import METADATA


def torque_loss_(outputs, targets, inputs, index=3):
    D = torch.from_numpy(METADATA['config_matrix']).to(torch.float).expand(outputs.shape[0], -1, -1).to(
        torch.device("cuda"))

    max_torque = METADATA['torque_range'][-1]

    high_limit_input = max_torque
    low_limit_input = -max_torque

    max_bias = METADATA['bias_range'][-1]

    # high_limit_label =  torch.tensor([1, 1, 1, max_bias, max_bias, max_bias]).to(torch.device("cuda"))
    # low_limit_label = torch.tensor([0, 0, 0, -max_bias, -max_bias, -max_bias]).to(torch.device("cuda"))

    high_limit_label = torch.tensor([1, 1, 1, max_bias, max_bias, max_bias]).to(torch.device("cuda"))
    low_limit_label = torch.tensor([0, 0, 0, -max_bias, -max_bias, -max_bias]).to(torch.device("cuda"))

    # print(inputs.shape, outputs.shape, targets.shape)
    command_torques = inputs[:, :, -3:]
    command_torques = command_torques * (high_limit_input - low_limit_input) + low_limit_input

    targets = targets * (high_limit_label - low_limit_label) + low_limit_label
    N_actuator = METADATA['N_actuator']
    eloss_label = targets[:, :, :N_actuator]
    bias_label = targets[:, :, N_actuator:]

    eloss_output = outputs[:, :, :N_actuator]
    bias_output = outputs[:, :, N_actuator:]

    total_loss = 0
    for i in range(int(METADATA['time_step_detection'] / METADATA['time_step_control'])):
        command_torque_i = command_torques[:, i, :].unsqueeze(1)
        torque_label = (torch.ones_like(eloss_label) - eloss_label) * command_torque_i + bias_label
        torque_output = (torch.ones_like(eloss_output) - eloss_output) * command_torque_i + bias_output

        # print("=" * 20)
        # print(eloss_label[1], command_torque_i[1], bias_label[1])
        # print(eloss_output[1], command_torque_i[1], bias_output[1])
        # print(torque_label[1], torque_output[1], torch.abs(torque_label - torque_output)[1])

        total_loss += (torque_label - torque_output) ** 2

    return total_loss.mean()


def torque_loss(outputs, targets, inputs, index=3):
    I = torch.tensor(METADATA['inertia_matrix'], dtype=torch.float32).to(torch.device("cuda"))
    expanded_I = I.unsqueeze(0).repeat(outputs.shape[0], 1, 1)

    max_torque = METADATA['torque_range'][-1]

    high_limit_omega = torch.tensor([0.32, 0.32, 0.32, 0.32, 0.32, 0.32]).to(torch.device("cuda"))
    low_limit_omega = torch.tensor([-0.32, -0.32, -0.32, -0.32, -0.32, -0.32]).to(torch.device("cuda"))

    high_limit_torque = max_torque
    low_limit_torque = -max_torque

    max_bias = METADATA['bias_range'][-1]

    # high_limit_label =  torch.tensor([1, 1, 1, max_bias, max_bias, max_bias]).to(torch.device("cuda"))
    # low_limit_label = torch.tensor([0, 0, 0, -max_bias, -max_bias, -max_bias]).to(torch.device("cuda"))

    high_limit_label = torch.tensor([1, 1, 1, max_bias, max_bias, max_bias]).to(torch.device("cuda"))
    low_limit_label = torch.tensor([0, 0, 0, -max_bias, -max_bias, -max_bias]).to(torch.device("cuda"))

    # print(inputs.shape, outputs.shape, targets.shape)
    omega = inputs[:, :, :-3]
    omega = omega * (high_limit_omega - low_limit_omega) + low_limit_omega
    omega_new, omega_old = omega[:, :, :3], omega[:, :, 3:6]

    command_torques = inputs[:, :, -3:]
    command_torques = command_torques * (high_limit_torque - low_limit_torque) + low_limit_torque

    targets = targets * (high_limit_label - low_limit_label) + low_limit_label
    N_actuator = METADATA['N_actuator']
    # eloss_label = targets[:, :, :N_actuator]
    # bias_label = targets[:, :, N_actuator:]

    eloss_output = outputs[:, :, :N_actuator]
    bias_output = outputs[:, :, N_actuator:]

    total_loss = 0
    for i in range(int(METADATA['time_step_detection'] / METADATA['time_step_control'])):
        omega_new_i = omega_new[:, i, :].unsqueeze(2)  # shape [B, 3, 1]
        omega_old_i = omega_old[:, i, :].unsqueeze(2)  # shape [B, 3, 1]

        part_left = torch.bmm(expanded_I, (omega_new_i - omega_old_i) / 0.1) + torch.cross(omega_old_i, torch.bmm(expanded_I , omega_old_i))

        command_torque_i = command_torques[:, i, :].unsqueeze(1)
        # torque_label = (torch.ones_like(eloss_label) - eloss_label) * command_torque_i + bias_label
        # torque_output = (torch.ones_like(eloss_output) - eloss_output) * command_torque_i + bias_output
        part_right = (torch.ones_like(eloss_output) - eloss_output) * command_torque_i + bias_output
        part_right = part_right.transpose(1,2)


        # print("=" * 20)
        # print(eloss_label[1], command_torque_i[1], bias_label[1])
        # print(eloss_output[1], command_torque_i[1], bias_output[1])
        # print(torque_label[1], torque_output[1], torch.abs(torque_label - torque_output)[1])

        total_loss += (part_left - part_right) ** 2

    return total_loss.mean()
