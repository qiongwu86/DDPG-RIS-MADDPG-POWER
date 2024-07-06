import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
# 数据定义
users = ['VU1', 'VU2', 'VU3', 'VU4', 'VU5', 'VU6', 'VU7', 'VU8']
values_group4 = [2.970, 3.058, 3.260, 2.979, 2.946, 3.035, 2.981, 8.342]
values_group5 = [2.970, 3.038, 2.974, 2.979, 2.946, 3.035, 2.981, 3.008]
values_group6 = [3.029, 3.108, 3.031, 3.009, 3.041, 3.089, 3.014, 3.071]

# 设置柱状图的宽度
bar_width = 0.25  # 调整宽度以容纳三组数据

# 设置位置
r4 = np.arange(len(users))
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]

# 创建柱状图
plt.bar(r4, values_group4, color='#FFD700', width=bar_width, edgecolor='grey', label='TD3')  # 金黄色
plt.bar(r5, values_group5, color='#FF6347', width=bar_width, edgecolor='grey', label='DDPG')  # 番茄色
plt.bar(r6, values_group6, color='#8A2BE2', width=bar_width, edgecolor='grey', label='SARL-MARL')  # 蓝紫色

# 添加标题和标签
plt.xlabel('VUs')
plt.ylabel('Buffer length')
#plt.title('Bar Chart Example with Three Groups')
plt.xticks([r + bar_width for r in range(len(users))], users)

# 添加图例
plt.legend()
plt.savefig('user_data.pdf', dpi=300, format='pdf')
# 显示图表
plt.show()