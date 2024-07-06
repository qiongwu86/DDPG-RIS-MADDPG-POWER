import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 字体设置
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# 定义移动平均函数
def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

# 参数
n_episode = 1000
window_size2 = 1  # 平滑窗口大小，你可以根据需要调整
x2 = np.linspace(0, n_episode, n_episode - window_size2 + 1, dtype=int)

# 加载和平滑数据
reward_maddpg0 = moving_average(np.load('3-User0_Reward_1000.npy'), window_size2)
reward_maddpg1 = moving_average(np.load('3-User1_Reward_1000.npy'), window_size2)
reward_maddpg2 = moving_average(np.load('3-User2_Reward_1000.npy'), window_size2)
reward_maddpg3 = moving_average(np.load('3-User3_Reward_1000.npy'), window_size2)
reward_maddpg4 = moving_average(np.load('3-User4_Reward_1000.npy'), window_size2)
reward_maddpg5 = moving_average(np.load('3-User5_Reward_1000.npy'), window_size2)
reward_maddpg6 = moving_average(np.load('3-User6_Reward_1000.npy'), window_size2)
reward_maddpg7 = moving_average(np.load('3-User7_Reward_1000.npy'), window_size2)

# 设置颜色和线型
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# 创建图表
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(x2, reward_maddpg0, color=colors[0], linestyle=linestyles[0])
plt.xlabel('Episode')
plt.ylabel('User1 Reward')

plt.subplot(2, 2, 2)
plt.plot(x2, reward_maddpg1, color=colors[1], linestyle=linestyles[1])
plt.xlabel('Episode')
plt.ylabel('User2 Reward')

plt.subplot(2, 2, 3)
plt.plot(x2, reward_maddpg2, color=colors[2], linestyle=linestyles[2])
plt.xlabel('Episode')
plt.ylabel('User3 Reward')

plt.subplot(2, 2, 4)
plt.plot(x2, reward_maddpg3, color=colors[3], linestyle=linestyles[3])
plt.xlabel('Episode')
plt.ylabel('User4 Reward')

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # 调整顶部间距
plt.savefig('user_reward1.pdf')
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(x2, reward_maddpg4, color=colors[4], linestyle=linestyles[4])
plt.xlabel('Episode')
plt.ylabel('User5 Reward')

plt.subplot(2, 2, 2)
plt.plot(x2, reward_maddpg5, color=colors[5], linestyle=linestyles[5])
plt.xlabel('Episode')
plt.ylabel('User6 Reward')

plt.subplot(2, 2, 3)
plt.plot(x2, reward_maddpg6, color=colors[6], linestyle=linestyles[6])
plt.xlabel('Episode')
plt.ylabel('User7 Reward')

plt.subplot(2, 2, 4)
plt.plot(x2, reward_maddpg7, color=colors[7], linestyle=linestyles[7])
plt.xlabel('Episode')
plt.ylabel('User8 Reward')  # 修改了标题，以确保唯一性

plt.tight_layout()
plt.subplots_adjust(top=0.95)  # 调整顶部间距
plt.savefig('D:\QKW\SARL_MARL\user_reward2.pdf')
plt.show()