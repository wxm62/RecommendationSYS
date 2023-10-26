import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f1_1 = [3,3,3,2,1]
f1_2 = [7,6,7,2,1]
f1_3 = [7,8,7,3,1]
f1_4 = [9,8,9,3,1]

x = ['w=0', 'w=0.5', 'w=1','w=1.5','w=2']
x_len = np.arange(len(x))
total_width, n = 1, 5
width = 0.2
xticks = x_len - (total_width - width) / 4
plt.figure(figsize=(15, 12), dpi=200)

ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)
plt.bar(xticks, f1_1, width=0.9 * width, label="Attention weights", color="#92a6be", edgecolor='black', linewidth=2,
        zorder=10)
plt.bar(xticks + width, f1_2, width=0.9 * width, label="Official", color="#c48d60", edgecolor='black', linewidth=2,
        zorder=10)
plt.text(xticks[0], f1_1[0] + 0.3, f1_1[0], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[1], f1_1[1] + 0.3, f1_1[1], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
plt.text(xticks[2], f1_1[2] + 0.3, f1_1[2], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)

plt.text(xticks[0] + width, f1_2[0] + 0.3, f1_2[0], ha='center', fontproperties='Times New Roman', fontsize=35,
         zorder=10)
plt.text(xticks[1] + width, f1_2[1] + 0.3, f1_2[1], ha='center', fontproperties='Times New Roman', fontsize=35,
         zorder=10)
plt.text(xticks[2] + width, f1_2[2] + 0.3, f1_2[2], ha='center', fontproperties='Times New Roman', fontsize=35,
         zorder=10)

plt.legend(prop={'family': 'Times New Roman', 'size': 35}, ncol=2)
x_len = [-0.1, 0.9, 1.9]
x_len = np.array(x_len)
plt.xticks(x_len, x, fontproperties='Times New Roman', fontsize=40)
plt.yticks(fontproperties='Times New Roman', fontsize=40)
plt.ylim(ymin=75)
plt.xlabel("Datasets", fontproperties='Times New Roman', fontsize=45)
plt.ylabel("Accuracy (%)", fontproperties='Times New Roman', fontsize=45)
ax.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['left'].set_color('black')

plt.show()
