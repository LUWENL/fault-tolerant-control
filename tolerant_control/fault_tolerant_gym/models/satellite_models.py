import numpy as np
import torch

from metadata import Initial_state, Fault_time, Fault_value, Double_Fault_time, Double_Fault_value
from tolerant_control.fault_tolerant_gym.models.attitude_maneuver import *
from tolerant_control.fault_tolerant_gym.models.dynamics_and_kinematics import *
from fault_detection.models.FaultNet import FaultNet


class Satellite(object):
    def __init__(self, scenario_id=None, fault_time=None, double_fault_time=None, end_time=METADATA['end_time'],
                 time_step=METADATA['time_step_control'],
                 mode=METADATA['mode'], N_actuator=METADATA['N_actuator'], config_matrix=METADATA['config_matrix'],
                 max_omega=METADATA['max_omega'], torque_range=METADATA['torque_range']):

        self.scenario_id = scenario_id
        self.mode = mode

        self.time_step = time_step
        self.current_time = - self.time_step
        self.end_time = end_time

        # inertia matrix
        self.I = METADATA["inertia_matrix"]

        # configuration matrix, shape: [3, N_actuator]
        self.N_actuator = N_actuator
        self.config_matrix = config_matrix
        self.fault_time = fault_time
        self.double_fault_time = double_fault_time
        self.torque_range = torque_range

        # parameters of actuator fault, shape: [N_actuator], [N_actuator]
        self.effectiveness_loss = np.zeros(shape=[self.N_actuator])
        self.additive_bias = np.zeros(shape=[self.N_actuator])
        # estimated parameters (marked as hat)
        self.effectiveness_loss_hat = np.zeros(shape=[self.N_actuator])
        self.additive_bias_hat = np.zeros(shape=[self.N_actuator])

        self.max_omega = max_omega
        self.omega = None  # the current angular velocity
        self.omega_with_noise = None

        self.quaternion0 = None
        self.desired_quaternion = np.array([0, 0, 0, 1])
        self.error_quaternion = None

        # for fault detection and identification
        if self.mode != 'generation':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = FaultNet(configs="default")
            self.model.load_state_dict(
                torch.load("fault_detection/checkpoints/FaultNet/best_model.pth", weights_only=True))

            # self.model = LSTMModel(configs="default")
            # self.model.load_state_dict(
            #     torch.load("fault_detection/checkpoints/LSTM/best_model.pth", weights_only=True))

            self.model.to(self.device)
            self.model.eval()

        # for record
        self.omega_list = []
        self.omega_n_list = []
        self.quaternion0_list = []
        self.error_quaternion_list = []
        self.control_torque_list = []
        self.fault_info_list = []
        self.fault_hat_info_list = []

    def reset(self):
        self.current_time = - self.time_step
        # parameters of actuator fault, shape: [N_actuator]
        self.effectiveness_loss = np.zeros(shape=[self.N_actuator])
        self.additive_bias = np.zeros(shape=[self.N_actuator])
        # estimated parameters (marked as hat)
        self.effectiveness_loss_hat = np.zeros(shape=[self.N_actuator])
        self.additive_bias_hat = np.zeros(shape=[self.N_actuator])

        # for record
        self.omega_list = []
        self.omega_n_list = []
        self.quaternion0_list = []
        self.error_quaternion_list = []
        self.control_torque_list = []
        self.fault_info_list = []
        self.fault_hat_info_list = []

        # initial attitude and fault
        if self.mode == 'generation':
            self.quaternion0 = generate_random_quaternion()
            self.omega = generate_random_omega(max_omega=METADATA['max_omega'])

        elif self.mode == 'train':
            self.quaternion0 = generate_random_quaternion()
            self.omega = generate_random_omega(max_omega=METADATA['max_omega'] * 0.2)
            # self.fault_time = np.random.randint(0, self.end_time * 0.2 / self.time_step) * self.time_step
            self.fault_time = np.random.randint( self.end_time * 0.2 / self.time_step, self.end_time / self.time_step) * self.time_step

        elif self.mode == 'test':
            self.quaternion0, self.omega = Initial_state["Test{}".format(str(self.scenario_id))]
            self.fault_time = Fault_time["Test{}".format(str(self.scenario_id))]

            if self.scenario_id in [2, 3, 4]:
                self.double_fault_time = Double_Fault_time["Test{}".format(str(self.scenario_id))]

        if METADATA['with_noise']:
            self.omega_with_noise = self.omega + np.random.normal(0, METADATA['sigma_ratio'] * np.abs(self.omega),
                                                                  size=3)
        else:
            self.omega_with_noise = self.omega

        # for record
        self.quaternion0_list.append(self.quaternion0)
        self.omega_list.append(self.omega)
        self.omega_n_list.append(self.omega_with_noise)
        self.control_torque_list.append([0 for i in range(self.N_actuator)])
        self.fault_info_list.append(np.concatenate([np.array([0, 0, 0]), np.array([0, 0, 0])]))
        self.fault_hat_info_list.append(np.concatenate([np.array([0, 0, 0]), np.array([0, 0, 0])]))

    def update_attitude(self, command_torque):

        # the external disturbance
        disturbance = generate_external_disturbance(t=self.current_time, omega=self.omega, constant=False)

        actual_tau = (np.ones_like(
            self.effectiveness_loss) - self.effectiveness_loss) * command_torque + self.additive_bias

        tau = actual_tau + disturbance
        # tau = self.config_matrix @ actual_tau + disturbance

        # integrate
        new_omega = integrate_omega(self.omega, self.time_step, tau, self.I)
        new_quaternion = integrate_quaternion(self.quaternion0, self.time_step, new_omega)

        # update the current attitude
        old_omega = self.omega
        self.quaternion0 = new_quaternion
        self.omega = new_omega
        if METADATA['with_noise']:
            self.omega_with_noise = (self.omega +
                                     np.random.normal(0, METADATA['sigma_ratio'] * np.abs(self.omega), size=3) +
                                     np.random.normal(0, METADATA['sigma_ratio'] / 10 * np.abs(old_omega), size=3))
        else:
            self.omega_with_noise = self.omega

    def get_desired_and_error_quaternion(self):

        conjugate_desired_quaternion = get_conjugate_quaternion(self.desired_quaternion)
        self.error_quaternion = quaternion_multiply(conjugate_desired_quaternion, self.quaternion0)

        # for record
        self.error_quaternion_list.append(self.error_quaternion)

        return self.desired_quaternion, self.error_quaternion

    def fault(self):

        if self.mode == 'train':
            self.effectiveness_loss = np.random.uniform(0, 0.5,
                                                        METADATA['N_actuator'])
            self.additive_bias = np.random.uniform(-0.15, 0.15,
                                                   METADATA['N_actuator'])

        elif self.mode == 'test':
            fault_info = Fault_value["Test{}".format(str(self.scenario_id))]
            self.effectiveness_loss = fault_info[:METADATA['N_actuator']]
            self.additive_bias = fault_info[-METADATA['N_actuator']:]

    def double_fault(self):
        double_fault_info = Double_Fault_value["Test{}".format(str(self.scenario_id))]
        self.effectiveness_loss = double_fault_info[:METADATA['N_actuator']]
        self.additive_bias = double_fault_info[-METADATA['N_actuator']:]

    def estimate_fault(self, verbose=False):

        # [Q_0, omega, control_torque]
        length = int(METADATA['time_step_detection'] / METADATA['time_step_control'])

        max_torque = METADATA['torque_range'][1]
        high_limit_input = np.array([0.75, 0.85, 0.85, max_torque, max_torque, max_torque])
        low_limit_input = - high_limit_input

        min_eloss, max_eloss = METADATA['eloss_range']
        min_bias, max_bias = METADATA['bias_range']
        high_limit_output = np.array([max_eloss, max_eloss, max_eloss, max_bias, max_bias, max_bias])
        low_limit_output = np.array([min_eloss, min_eloss, min_eloss, min_bias, min_bias, min_bias])

        if len(self.omega_n_list) >= length + 1 and len(self.control_torque_list) >= length:
            time_series = []
            latest_omega_n_list = self.omega_n_list[-(length + 1):]
            latest_control_torque_list = self.control_torque_list[-length:]

            for i in range(1, length + 1):
                omega_new_n = latest_omega_n_list[i]
                omega_old_n = latest_omega_n_list[i - 1]
                control_torque = latest_control_torque_list[i - 1]

                part_left = self.I @ (omega_new_n - omega_old_n) / METADATA['time_step_control'] + np.cross(omega_old_n,
                                                                                                            self.I @ omega_old_n)

                time_series.append(
                    np.concatenate(
                        [part_left, control_torque])
                )

            time_series = np.array(time_series)

            # norm
            time_series = (time_series - low_limit_input) / (high_limit_input - low_limit_input)
            time_series = torch.tensor(time_series, dtype=torch.float32).unsqueeze(dim=0).to(self.device)

            # de-norm
            output = self.model(time_series).detach().cpu().numpy().reshape(6)
            estimated_fault = output * (high_limit_output - low_limit_output) + low_limit_output

        else:
            estimated_fault = np.array([0, 0, 0, 0, 0, 0])

        self.effectiveness_loss_hat = estimated_fault[:3]
        self.additive_bias_hat = estimated_fault[-3:]

        label = np.concatenate([self.effectiveness_loss, self.additive_bias])

        if verbose:
            print(repr(estimated_fault))
            print(repr(label))
            print(np.mean((estimated_fault - label) ** 2))
            print("-" * 25)

        return estimated_fault

    def update(self, control_torque):

        self.current_time += self.time_step

        if self.mode != 'generation':
            # avoid float error
            if self.fault_time < self.current_time < self.fault_time + self.time_step:
                self.fault()

            if self.double_fault_time is not None:
                if self.double_fault_time < self.current_time < self.double_fault_time + self.time_step:
                    self.double_fault()

            if self.current_time >= (self.end_time * 0.2):
                if self.mode == 'test':
                    self.estimate_fault()



        # update attitude and state (t -> t+1)
        self.update_attitude(control_torque)

        # for record
        self.quaternion0_list.append(self.quaternion0)
        self.omega_list.append(self.omega)
        self.omega_n_list.append(self.omega_with_noise)
        self.control_torque_list.append(control_torque)
        self.fault_info_list.append(np.concatenate([self.effectiveness_loss, self.additive_bias]))
        self.fault_hat_info_list.append(np.concatenate([self.effectiveness_loss_hat, self.additive_bias_hat]))

    def observation(self):
        # state = [Q_e, omega, additive_bias_hat]

        desired_quaternion, error_quaternion = self.get_desired_and_error_quaternion()

        if self.mode == 'train':
            return np.concatenate(
                [error_quaternion, self.omega_with_noise, self.effectiveness_loss, self.additive_bias]).astype(
                np.float32)

        elif self.mode == 'test':
            return np.concatenate(
                [error_quaternion, self.omega_with_noise, self.effectiveness_loss_hat, self.additive_bias_hat]).astype(
                np.float32)

    def reward(self):
        # maneuver reward
        _, _, _, qe_4 = self.error_quaternion
        maneuver_reward = qe_4

        maneuver_weight = 10 if qe_4 >= 0.98 else 1

        # progress reward
        progress_reward = 0
        if len(self.error_quaternion_list) >= 2:
            if qe_4 >= self.error_quaternion_list[-2][-1]:
                progress_reward += 5

        return maneuver_weight * maneuver_reward + progress_reward
