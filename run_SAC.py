import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from metadata import METADATA
from utils import plot_quaternion, plot_omega, plot_torque, plot_ua, plot_estimation_error, \
    plot_el, plot_euler_angle
import numpy as np
from tolerant_control.fault_tolerant_gym.envs.fault_tolerant import FaultTolerantEnv

register(
    id="FaultTolerantEnv-v0",
    entry_point="tolerant_control.fault_tolerant_gym.envs:FaultTolerantEnv",
)

env = gym.make('FaultTolerantEnv-v0')

mode = METADATA['mode']
seed = METADATA['seed']

if mode == 'train':
    model = SAC("MlpPolicy", env, verbose=1, seed=METADATA['seed'])
    model.learn(total_timesteps=int(1e6), progress_bar=True)
    model.save("SAC_FTC_" + str(seed))
    del model

model = SAC.load("SAC_FTC_" + str(seed), env=env)

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()

# for plot
plot_dict = {'Current quaternion': [],
             'Desired quaternion': [],
             'Error quaternion': [],
             'Current Omega': [],
             'Command Torque': [],
             'True Fault Info': [],
             'Estimated Fault Info': [],
             }

dones = [False]
while not dones[0]:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    # print(obs, action)

    # for plot
    info = info[0]

    plot_dict['Desired quaternion'].append(info['Desired quaternion'])
    plot_dict['Error quaternion'].append(info['Error quaternion'])
    plot_dict['Current quaternion'].append(info['Current quaternion'])
    plot_dict['Current Omega'].append(info['Current Omega'])
    plot_dict['Command Torque'].append(info['Command Torque'])
    plot_dict['True Fault Info'].append(info['True Fault Info'])
    plot_dict['Estimated Fault Info'].append(info['Estimated Fault Info'])


qe_4_list = np.array(plot_dict['Error quaternion'])
np.set_printoptions(threshold=np.inf)
print(repr(qe_4_list[:, -1]))
# print(mean_reward)

print(f"Number of qe_4 greater than  0.995: {np.sum(qe_4_list >=  0.995)}")
print(f"Number of qe_4 greater than  0.999: {np.sum(qe_4_list >=  0.999)}")


def find_threshold_index(qe_4_list, threshold=0.995):

    n = len(qe_4_list)
    for i in range(n):
        # 检查从当前索引开始的所有值是否都大于阈值
        if all(x > threshold for x in qe_4_list[i:]):
            return i
    return -1  # 如果没有找到满足条件的时刻，返回 -1


print(f"从时刻 {find_threshold_index(qe_4_list[:, -1], 0.995) / 10} 开始，之后的 qe_4 值都大于 0.995。")
print(f"从时刻 {find_threshold_index(qe_4_list[:, -1], 0.999) / 10} 开始，之后的 qe_4 值都大于 0.999。")


# # plot_quaternion(plot_dict['Desired quaternion'], 'Desired quaternion')
# plot_quaternion(plot_dict['Error quaternion'], 'Error quaternion')
plot_quaternion(plot_dict['Current quaternion'], 'Current quaternion')
# plot_euler_angle(plot_dict['Current quaternion'], 'Current Euler Angle')
plot_omega(plot_dict['Current Omega'], 'Current Angular Velocity')
# plot_torque(plot_dict['Command Torque'], 'Command Torque')

plot_el(plot_dict['True Fault Info'], 'True Effectiveness Loss')
plot_ua(plot_dict['True Fault Info'], 'True Additive Bias Truth')
plot_el(plot_dict['Estimated Fault Info'], 'Estimated Effectiveness Loss')
plot_ua(plot_dict['Estimated Fault Info'], 'Estimated Additive Bias')
# plot_estimation_error(plot_dict['True Fault Info'], 'Fault Estimation Error')
