import matplotlib.pyplot as plt
import numpy as np
import os
from metadata import METADATA, Fault_time, Double_Fault_time
import seaborn as sns
import pandas as pd


def mkdir(scenario_id=METADATA['scenario_id']):
    pair_name = 'Scenario' + str(scenario_id)
    if not os.path.exists(os.path.join('figures', pair_name)):
        os.makedirs(os.path.join('figures', pair_name))

    return os.path.join('figures', pair_name)


def plot_quaternion(data_list, data_name, fault_times=None):
    save_path = mkdir()
    time_list = np.array([i * METADATA['time_step_control'] for i in range(len(data_list))])

    q_1_list = []
    q_2_list = []
    q_3_list = []
    q_4_list = []

    for data in data_list:
        q_1, q_2, q_3, q_4 = data
        q_1_list.append(q_1)
        q_2_list.append(q_2)
        q_3_list.append(q_3)
        q_4_list.append(q_4)

    plt.plot(time_list, q_1_list, label='$q_1$')
    plt.plot(time_list, q_2_list, label='$q_2$')
    plt.plot(time_list, q_3_list, label='$q_3$')
    plt.plot(time_list, q_4_list, label='$q_4$')

    plt.ylim(-0.2, 1.1)

    if fault_times == None:
        fault_times = [Fault_time["Test{}".format(METADATA['scenario_id'])]]
    for fault_time in fault_times:
        plt.axvline(x=fault_time, color='red', linestyle=':')
        plt.text(fault_time + 15, 0.5, 'Fault', color='red', rotation=0, ha='center', va='bottom')

    if METADATA['scenario_id'] in [2, 3, 4]:
        double_fault_time = Double_Fault_time["Test{}".format(METADATA['scenario_id'])]
        plt.axvline(x=double_fault_time, color='red', linestyle=':')
        plt.text(double_fault_time + 15, 0.5, 'Fault', color='red', rotation=0, ha='center', va='bottom')

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name)

    plt.legend(loc='right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()

def quaternion_to_euler_xyz(q):
    q1, q2, q3, q4 = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q4 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1**2 + q2**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q4 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q4 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2**2 + q3**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg

def plot_euler_angle(q_list, data_name):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(q_list))])
    roll_list = []
    pitch_list = []
    yaw_list = []

    for q in q_list:
        roll, pitch, yaw = quaternion_to_euler_xyz(q)
        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)

    plt.plot(time_list, roll_list, label='Roll (X)')
    plt.plot(time_list, pitch_list, label='Pitch (Y)')
    plt.plot(time_list, yaw_list, label='Yaw (Z)')

    plt.ylim(-80, 80)
    plt.xlabel('Time, s')
    plt.ylabel('Euler Angle, deg')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)
    plt.close()


def plot_omega(data_list, data_name, data_unit='rad/s'):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(data_list))])

    omega_x_list = []
    omega_y_list = []
    omega_z_list = []

    for data in data_list:
        omega_x, omega_y, omega_z = data
        omega_x_list.append(omega_x)
        omega_y_list.append(omega_y)
        omega_z_list.append(omega_z)

    plt.plot(time_list, omega_x_list, label='$\omega_x$')
    plt.plot(time_list, omega_y_list, label='$\omega_y$')
    plt.plot(time_list, omega_z_list, label='$\omega_z$')

    plt.ylim(-METADATA['max_omega'], METADATA['max_omega'])

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name + ', ' + data_unit)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


def plot_torque(data_list, data_name='command torque', data_unit='Nm'):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(data_list))])

    torque_x_list = []
    torque_y_list = []
    torque_z_list = []

    for data in data_list:
        torque_x, torque_y, torque_z = data
        torque_x_list.append(torque_x)
        torque_y_list.append(torque_y)
        torque_z_list.append(torque_z)

    plt.plot(time_list, torque_x_list, label='$torque_x$')
    plt.plot(time_list, torque_y_list, label='$torque_y$')
    plt.plot(time_list, torque_z_list, label='$torque_z$')

    # plt.ylim(METADATA['torque_range'][0] * 1.5, METADATA['torque_range'][1] * 1.5)

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name + ', ' + data_unit)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


def plot_el(data_list, data_name='Estimated Error', data_unit=''):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(data_list))])

    el_hat_x_list = []
    el_hat_y_list = []
    el_hat_z_list = []

    for data in data_list:
        el_hat_x, el_hat_y, el_hat_z, _, _, _ = data

        el_hat_x_list.append(el_hat_x)
        el_hat_y_list.append(el_hat_y)
        el_hat_z_list.append(el_hat_z)

    plt.plot(time_list, el_hat_x_list, label='$\hat{el}_{1}$')
    plt.plot(time_list, el_hat_y_list, label='$\hat{el}_{2}$')
    plt.plot(time_list, el_hat_z_list, label='$\hat{el}_{3}$')

    plt.ylim(METADATA['eloss_range'][0], METADATA['eloss_range'][1])

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name + ', ' + data_unit)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


def plot_ua(data_list, data_name='Estimated Error', data_unit='Nm'):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(data_list))])

    ua_hat_x_list = []
    ua_hat_y_list = []
    ua_hat_z_list = []

    for data in data_list:
        _, _, _, ua_hat_x, ua_hat_y, ua_hat_z = data

        ua_hat_x_list.append(ua_hat_x)
        ua_hat_y_list.append(ua_hat_y)
        ua_hat_z_list.append(ua_hat_z)

    plt.plot(time_list, ua_hat_x_list, label='$\hat{u}_{a_1}$')
    plt.plot(time_list, ua_hat_y_list, label='$\hat{u}_{a_2}$')
    plt.plot(time_list, ua_hat_z_list, label='$\hat{u}_{a_3}$')

    plt.ylim(METADATA['bias_range'][0], METADATA['bias_range'][1])
    # plt.ylim(METADATA['torque_range'][0] , METADATA['torque_range'][1])

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name + ', ' + data_unit)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()


def plot_estimation_error(data_list, data_name='Fault Estimation Error', data_unit='Nm'):
    save_path = mkdir()
    time_list = np.array([i * METADATA["time_step_control"] for i in range(len(data_list))])

    error_x_list = []
    error_y_list = []
    error_z_list = []

    for data in data_list:
        ua_x, ua_y, ua_z, ua_hat_x, ua_hat_y, ua_hat_z = data

        # error_x_list.append(np.abs(ua_x - ua_hat_x))
        # error_y_list.append(np.abs(ua_y - ua_hat_y))
        # error_z_list.append(np.abs(ua_z - ua_hat_z))
        error_x_list.append(ua_x - ua_hat_x)
        error_y_list.append(ua_y - ua_hat_y)
        error_z_list.append(ua_z - ua_hat_z)

    plt.plot(time_list, error_x_list, label='$error_1$')
    plt.plot(time_list, error_y_list, label='$error_2$')
    plt.plot(time_list, error_z_list, label='$error_3$')

    plt.ylim(METADATA['torque_range'][0] * 0.01, METADATA['torque_range'][1] * 0.01)

    plt.xlabel('Time' + ', s')
    plt.ylabel(data_name + ', ' + data_unit)

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, data_name + '.pdf'), dpi=500)

    plt.close()



def plot_distribution(data):
    # Convert to a DataFrame for easier plotting
    columns = ['$e_1$', '$e_2$', '$e_3$', '$u_{a1}$', '$u_{a2}$', '$u_{a3}$']
    df = pd.DataFrame(data, columns=columns)

    # Set style for plots
    sns.set(style="whitegrid")

    # Plot histograms for e1, e2, e3
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['$e_1$', '$e_2$', '$e_3$']):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[col], kde=True, bins=50, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('e_dist.pdf', dpi=500)  # Save as PDF with DPI=100
#     plt.show()

    # Plot histograms for ua1, ua2, ua3
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['$u_{a1}$', '$u_{a2}$', '$u_{a3}$']):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[col], kde=True, bins=50, color='green')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xlim(-0.15, 0.15)
    plt.tight_layout()
    plt.savefig('ua_dist.pdf', dpi=500)  # Save as PDF with DPI=100
#     plt.show()
