import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

n_episode = 1000
window_size1 = 1

reward_maddpg = moving_average(np.load('3-Reward_MADDPG_1000.npy'), window_size1)
TD3 = moving_average(np.load('3-Reward_TD3_1000.npy'), window_size1)
DDPG = moving_average(np.load('3-Reward_DDPG_1000.npy'), window_size1)

x1 =np.linspace(0,n_episode, n_episode, dtype=int)

plt.figure(1)
#plt.plot(x1, reward_ddpg, label='DDPG')
plt.plot(x1, reward_maddpg, label='Proposed Algorithm', color='red')
plt.plot(x1, TD3, label='TD3', color='salmon')
plt.plot(x1, DDPG, label='DDPG', color='skyblue')
#plt.plot(x1, reward_sac, label='SAC')
plt.grid(True, linestyle='-', linewidth=0.5)
# plt.yticks(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(bbox_to_anchor=(0.6, 0.3), fontsize=14)
plt.savefig('D:\QKW\SARL_MARL\Reward.pdf', dpi=300, format='pdf')
plt.show()