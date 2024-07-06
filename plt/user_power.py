import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
# 数据定义
users = ['VU1', 'VU2', 'VU3', 'VU4', 'VU5', 'VU6', 'VU7', 'VU8']
values_group1 = [0.547, 0.132, 0.063, 0.193, 0.646, 0.277, 0.289, 0.033]
values_group2 = [0.962, 1.009, 1.005, 1.122, 0.943, 0.843, 1.003, 0.927]
values_group3 = [0.081, 0.081, 0.081, 0.084, 0.079, 0.098, 0.099, 0.096]

# 设置柱状图的宽度
bar_width = 0.25

# 设置位置
r1 = np.arange(len(users))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 创建柱状图
plt.bar(r1, values_group1, color='#FF9999', width=bar_width, edgecolor='grey', label='TD3')  # 浅红色
plt.bar(r2, values_group2, color='#66B3FF', width=bar_width, edgecolor='grey', label='DDPG')  # 浅蓝色
plt.bar(r3, values_group3, color='#99FF99', width=bar_width, edgecolor='grey', label='SARL-MARL')  # 浅绿色

# 添加标题和标签
plt.xlabel('VUs')
plt.ylabel('Power')
#plt.title('Bar Chart Example')
plt.xticks([r + bar_width for r in range(len(users))], users)
# 调整y轴范围
plt.ylim(0, 1.5)  # 根据数据范围设置合适的y轴范围
# 添加图例
plt.legend()
plt.savefig('user_power.pdf', dpi=300, format='pdf')
# 显示图表
plt.show()