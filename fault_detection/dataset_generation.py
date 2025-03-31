from tolerant_control.fault_tolerant_gym.models.satellite_models import Satellite
from metadata import METADATA
import numpy as np
import pandas as pd


def fault_infos_generation(length, no_fault_ratio=0.2):
    fault_infos = []

    for i in range(length):
        effectiveness_loss = np.random.uniform(METADATA['eloss_range'][0], METADATA['eloss_range'][1],
                                               METADATA['N_actuator'])
        additive_bias = np.random.uniform(METADATA['bias_range'][0], METADATA['bias_range'][1],
                                          METADATA['N_actuator'])
        if i <= length * (1 - no_fault_ratio):
            fault_infos.append([effectiveness_loss, additive_bias])
        else:
            fault_infos.append([np.zeros(METADATA['N_actuator']), np.zeros(METADATA['N_actuator'])])

    return fault_infos


def dataset_generation(sat, fault_info, torque_range):
    sat.reset()
    sat.current_time = np.random.randint(0, METADATA['end_time'])

    # record the initial data
    time_series = []
    sat.effectiveness_loss = fault_info[0]
    sat.additive_bias = fault_info[1]


    for current_time in range(int(sat.end_time / sat.time_step)):
        omega_old_noise = sat.omega_with_noise

        control_torque = np.random.uniform(torque_range[0], torque_range[1], sat.N_actuator)

        sat.update(control_torque)

        omega_new_noise = sat.omega_with_noise

        part_left = sat.I @ (omega_new_noise - omega_old_noise) / METADATA['time_step_control'] + np.cross(
            omega_old_noise, sat.I @ omega_old_noise)


        time_series.append(np.concatenate([part_left, control_torque]))  # len = 3 + 3 = 6

    time_series = np.array(time_series)

    return time_series, np.concatenate([sat.effectiveness_loss, sat.additive_bias])[np.newaxis, :]


if __name__ == '__main__':

    type_of_set = "train"
    all_length = int(2e5)
    # type_of_set = "test"
    # all_length = int(1e4)

    inputs = []
    targets = []

    # generate fault in advance
    fault_infos = fault_infos_generation(length=all_length)

    sat = Satellite(mode='generation', end_time=METADATA['time_step_detection'],
                    time_step=METADATA['time_step_control'])

    print("Start Simulation!")

    for l in range(all_length):
        # input: [Q_{t-1}, \omega_{t-1}, \tau_{t-1}, Q_{t}, \omega_{t}]
        # target:[    e1, e2, ..., e_{N_{actuators}}, u_{a1}, u_{a2}, ..., u_{a_{N_{actuators}}}    ]
        if l % 1000 == 0:
            print("Finish {}-th simulation".format(l))
        input, target = dataset_generation(sat, fault_info=fault_infos[l],
                                           torque_range=METADATA['torque_range'])

        # print(repr(input), repr(target))
        # print(input.shape, target.shape)

        inputs.append(input)
        targets.append(target)

    inputs = np.array(inputs).reshape(-1, 3 + METADATA['N_actuator'])
    targets = np.array(targets).reshape(-1, 2 * METADATA['N_actuator'])

    print(inputs.shape, targets.shape)

    np.savez('dataset/' + type_of_set + '_data.npz', inputs=inputs, targets=targets)

    print("Saving successfully done!")
