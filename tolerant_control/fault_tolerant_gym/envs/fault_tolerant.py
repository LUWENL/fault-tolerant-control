import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
from tolerant_control.fault_tolerant_gym.models.satellite_models import Satellite
from metadata import METADATA
from utils import plot_quaternion, plot_el, plot_ua, plot_omega


class FaultTolerantEnv(gym.Env):
    metadata = {}

    def __init__(self, mode=METADATA['mode'], time_step_control=METADATA['time_step_control'],
                 time_step_detection=METADATA['time_step_detection']):
        # train or test
        self.mode = mode

        # get the id of scenarios
        self.scenario_id = None
        if self.mode == 'test':
            self.scenario_id = METADATA["scenario_id"]

        # time
        self.time_step_control = time_step_control
        self.time_step_detection = time_step_detection

        # agent and target
        self.agent = Satellite(scenario_id=self.scenario_id, mode=self.mode)

        # limit of quaternion
        quaternion_limit = np.array([1.0 for i in range(4)])
        # limit of omega
        self.max_omega = self.agent.max_omega
        max_omega = np.array([METADATA['max_omega'] for i in range(3)])

        # limit of bias
        el_high_limit = np.array([METADATA['eloss_range'][1] for i in range(self.agent.N_actuator)])
        el_low_limit = np.array([METADATA['eloss_range'][0] for i in range(self.agent.N_actuator)])
        bias_high_limit = np.array([METADATA["bias_range"][1] for i in range(self.agent.N_actuator)])
        bias_low_limit = np.array([METADATA["bias_range"][0] for i in range(self.agent.N_actuator)])

        high = np.array(np.concatenate([quaternion_limit, max_omega, el_high_limit, bias_high_limit]), dtype=np.float32)
        low = np.array(np.concatenate([-quaternion_limit, -max_omega, el_low_limit, bias_low_limit]), dtype=np.float32)

        shape = 4 + 3 + self.agent.N_actuator * 2  # Q, omega, eloss, bias
        self.observation_space = spaces.Box(low=low, high=high, shape=(shape,), dtype=np.float32)

        large_torque = 0.5
        medium_torque = 0.05
        little_torque = 0.005
        if METADATA['is_discrete']:
            self.action_space = spaces.Discrete(19)
            self._action_to_torques = {
                0: np.array([0, 0, 0]),
                1: np.array([large_torque, 0, 0]),
                2: np.array([-large_torque, 0, 0]),
                3: np.array([medium_torque, 0, 0]),
                4: np.array([-medium_torque, 0, 0]),
                5: np.array([little_torque, 0, 0]),
                6: np.array([-little_torque, 0, 0]),
                7: np.array([0, large_torque, 0]),
                8: np.array([0, -large_torque, 0]),
                9: np.array([0, medium_torque, 0]),
                10: np.array([0, -medium_torque, 0]),
                11: np.array([0, little_torque, 0]),
                12: np.array([0, -little_torque, 0]),
                13: np.array([0, 0, large_torque]),
                14: np.array([0, 0, -large_torque]),
                15: np.array([0, 0, medium_torque]),
                16: np.array([0, 0, -medium_torque]),
                17: np.array([0, 0, little_torque]),
                18: np.array([0, 0, -little_torque]),
            }
        else:
            self.action_space = spaces.Box(low=-large_torque, high=large_torque, shape=(self.agent.N_actuator,),
                                           dtype=np.float32)

    def step(self, action):

        # action = action[0]
        if METADATA['is_discrete']:
            torques = self._action_to_torques[action]
        else:
            torques = action

        # print(torques)
        # torques = np.array([0.2, 0.2, 0.2])
        # torques = np.random.uniform(-0.3, 0.3, 3)
        # print(torques)
        # print('-' * 20)


        # update the position and attitude of satellite
        self.agent.update(torques)

        reward = self.agent.reward()

        terminated = (self.agent.end_time < self.agent.current_time < self.agent.end_time + self.agent.time_step)

        # state for agent
        observation = self._get_obs()
        # info for developer
        info = self._get_info()

        # print(observation, reward, info['Detected Time'])
        # print('\n')

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=METADATA['seed'])

        # reset agent
        self.agent.reset()

        observation = self._get_obs()

        info = self._get_info()

        return observation, info

    def _get_obs(self):

        return self.agent.observation()

    def _get_info(self):
        info = {
            "Desired quaternion": self.agent.desired_quaternion,
            "Error quaternion": self.agent.error_quaternion_list[-1],
            "Current quaternion": self.agent.quaternion0_list[-1],

            "Current Omega": self.agent.omega_list[-1],
            "Command Torque": self.agent.control_torque_list[-1],
            "True Fault Info": self.agent.fault_info_list[-1],
            "Estimated Fault Info": self.agent.fault_hat_info_list[-1],
        }

        return info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    import gymnasium as gym

    vec_env = FaultTolerantEnv()
    obs = vec_env.reset()

    dones = [False]
    while not dones[0]:
        torque_range = METADATA['torque_range']
        action = np.random.uniform(torque_range[0], torque_range[1], 3)
        obs, rewards, dones[0], _, info = vec_env.step(action)
