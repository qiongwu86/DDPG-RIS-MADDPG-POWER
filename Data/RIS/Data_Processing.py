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
window_size1 = 10

ris16 = moving_average(np.load('3_Phase_Reward_DDPG_1000-16.npy'), window_size1)
ris36 = moving_average(np.load('3_Phase_Reward_DDPG_1000.npy'), window_size1)
ris64 = moving_average(np.load('3_Phase_Reward_DDPG_1000-64.npy'), window_size1)
ris100 = moving_average(np.load('3_Phase_Reward_DDPG_1000-100.npy'), window_size1)
x1 =np.linspace(0,n_episode, n_episode-window_size1+1, dtype=int)

plt.figure(1)
#plt.plot(x1, reward_ddpg, label='DDPG')
plt.plot(x1, ris16, label='N = 16', color='skyblue')
plt.plot(x1, ris36, label='N = 36', color='salmon')
plt.plot(x1, ris64, label='N = 64', color='blue')
plt.plot(x1, ris100, label='N = 100', color='red')
#plt.plot(x1, reward_sac, label='SAC')
plt.grid(True, linestyle='-', linewidth=0.5)
# plt.yticks(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='upper left', fontsize=14)
plt.savefig('D:\QKW\SARL_MARL\RIS_Reward.pdf', dpi=300, format='pdf')
plt.show()