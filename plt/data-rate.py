import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
# 数据
x = np.arange(1, 4.5, 0.5)
y6 = [8.03, 12, 16.237, 20.32, 24.4, 28.78, 33.23]
y7 = [8.285, 12.8, 17.5, 22.3677, 27.3255, 32.39, 37.38]
y8 = [8.04, 12.177, 16.463, 20.833, 25.407, 30, 34.65]
y9 = [8.01, 11.95, 16, 20, 23.93, 28, 32]
y10 = [8.02, 12, 16.175, 21.53, 29.57, 34.3, 750]  # 注意最后一个值显著增大

# 选择一个上限值
upper_limit = 50

# 截断数据
y10_truncated = [min(val, upper_limit) for val in y10]

# 创建画布
plt.figure()

# 绘制常规数据
plt.plot(x, y6, marker='o', linestyle='-', color='#1f77b4', label='SARL-MARL')
plt.plot(x, y7, marker='s', linestyle='--', color='#ff7f0e', label='NO-RIS')
plt.plot(x, y8, marker='^', linestyle='-.', color='#2ca02c', label='Random-RIS')
plt.plot(x, y9, marker='p', linestyle=':', color='#d62728', label='DDPG')

# 绘制截断后的数据
plt.plot(x, y10_truncated, marker='x', linestyle='--', color='#9467bd', label=f'TD3 (truncated at {upper_limit})')

# 添加网格线
plt.grid(True)

# 设置标签和标题
plt.xlabel('Task arrival rate / Mbps', fontsize=14)
plt.ylabel('Buffer length', fontsize=14)
#plt.title('Comparison of Buffer Lengths for Different Algorithms', fontsize=16)
plt.legend()
plt.savefig('D:\QKW\SARL_MARL\Buffer_rate.pdf', dpi=300, format='pdf')
# 显示图形
plt.tight_layout()
plt.show()