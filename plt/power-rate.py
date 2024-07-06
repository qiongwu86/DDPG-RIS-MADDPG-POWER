import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# 数据
x = np.arange(1, 4.5, 0.5)
y1 = [0.644, 0.6543, 0.66, 0.68, 0.699, 0.721, 0.768]
y2 = [0.6476, 0.6576, 0.676, 0.712, 0.7686, 0.8258, 0.898]
y3 = [0.647766, 0.654, 0.663, 0.6817, 0.72, 0.76, 0.81275]
y4 = [7.96, 7.6, 8.02, 7.67, 7.8, 7.8765, 7.82]
y5 = [2.186, 2.01, 2.25, 2.09, 2.18, 2.2466, 2.5]

# 创建一个包含原始数据和放大子图的画布
fig, ax = plt.subplots()

# 绘制原始数据，调整颜色方案
ax.plot(x, y1, marker='o', linestyle='-', color='#1f77b4', label='SARL-MARL')
ax.plot(x, y2, marker='s', linestyle='--', color='#ff7f0e', label='NO-RIS')
ax.plot(x, y3, marker='^', linestyle='-.', color='#2ca02c', label='Random-RIS')
ax.plot(x, y4, marker='p', linestyle=':', color='#d62728', label='DDPG')
ax.plot(x, y5, marker='x', linestyle='--', color='#9467bd', label='TD3')

# 添加放大的子图
zoomed_inset_ax = fig.add_axes([0.16, 0.42, 0.5, 0.4])  # 添加放大的子图的位置和大小
zoomed_inset_ax.plot(x, y1, marker='o', linestyle='-', color='#1f77b4', label='SARL-MARL')
zoomed_inset_ax.plot(x, y2, marker='s', linestyle='--', color='#ff7f0e', label='NO-RIS')
zoomed_inset_ax.plot(x, y3, marker='^', linestyle='-.', color='#2ca02c', label='Random-RIS')
zoomed_inset_ax.set_xlim(0.9, 4.1)  # 设置放大的子图 x 轴范围
zoomed_inset_ax.set_ylim(0.6, 0.91)  # 设置放大的子图 y 轴范围
zoomed_inset_ax.grid(True)

# 设置标签和标题
ax.set_xlabel('Task arrival rate / Mbps', fontsize=14)
ax.set_ylabel('Total power consumption', fontsize=14)
ax.legend()

# 添加网格线
ax.grid(True)

# 显示图形
plt.tight_layout()

plt.savefig('D:\QKW\SARL_MARL\Power_rate.pdf', dpi=300, format='pdf')
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
# 数据
x = np.arange(1, 4.5, 0.5)  # 横坐标从1到4，间隔为0.5
y1 = [0.648, 0.6528, 0.6587, 0.67, 0.699, 0.727, 0.77]
y2 = [0.6476, 0.6576, 0.676, 0.712, 0.7686, 0.8258, 0.898]
y3 = [0.647766, 0.654, 0.663, 0.6817, 0.72, 0.76, 0.81275]
y4 = [7.96, 7.6, 8.02, 7.67, 7.8, 7.8765, 7.82]
y5 = [2.186, 2.01, 2.25, 2.09, 2.18, 2.2466, 2.5]

# 绘图
plt.figure(figsize=(10, 6))

# 绘制折线，并设置颜色、线型、标记和标签
plt.plot(x, y1, marker='o', linestyle='-', color='blue', label='SARL-MARL')
plt.plot(x, y2, marker='s', linestyle='-', color='red', label='NO-RIS')
plt.plot(x, y3, marker='^', linestyle='-', color='green', label='Random-RIS')
plt.plot(x, y4, marker='d', linestyle='-', color='purple', label='DDPG')
plt.plot(x, y5, marker='x', linestyle='-', color='orange', label='TD3')

# 添加标题和标签
#plt.title('Comparison of Five Datasets')
plt.xlabel('Task arrival rate / Mbps',  fontsize=16)
plt.ylabel('Power consumption', fontsize=16)

# 显示图例，调整位置和样式
plt.legend(bbox_to_anchor=(0.25, 0.88))

# 显示网格，设置线型和透明度
plt.grid(True, linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()'''