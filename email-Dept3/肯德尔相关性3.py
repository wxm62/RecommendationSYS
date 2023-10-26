# Tau_b
from scipy.stats.stats import kendalltau
import numpy as np
import pandas as pd

# data = pd.read_csv('importance_dept3重要性从大到小.csv')
# dat1 = data['node'].array

# kshelldata = pd.read_csv('result-kshell2从大到小.csv')
# dat1 = kshelldata['nodeid'].array #-0.024514811031664963

# bcdata = pd.read_csv('result_bc从大到小.csv')
# dat1 = bcdata['node'].array #0.1041879468845761

ccdata = pd.read_csv('result_cc从大到小.csv')
dat1 = ccdata['node'].array # 0.0602655771195097

degreedata = pd.read_csv('result_degree从大到小.csv')
dat2 = degreedata['nodeid'].array  # 0.15321756894790603

# dat1 = np.array([3,5,1,6,7,2,8,8,4])
# dat2 = np.array([5,3,2,6,8,1,7,8,4])

c = 0
d = 0
t_x = 0
t_y = 0
for i in range(len(dat1)):
    for j in range(i + 1, len(dat1)):
        if (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) > 0:
            c = c + 1
        elif (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) < 0:
            d = d + 1
        else:
            if (dat1[i] - dat1[j]) == 0 and (dat2[i] - dat2[j]) != 0:
                t_x = t_x + 1
            elif (dat1[i] - dat1[j]) != 0 and (dat2[i] - dat2[j]) == 0:
                t_y = t_y + 1

tau_b = (c - d) / np.sqrt((c + d + t_x) * (c + d + t_y))

print('tau_b = {0}'.format(tau_b))
print('kendalltau(dat1,dat2) =  {0}'.format(kendalltau(dat1, dat2)))

# # #得到KSCNR方法的节点重要性排序
# data = pd.read_csv('importance_dept3重要性从大到小.csv')
# x = data['node'].array
# print(x)
# # [60, 49, 25, 85, 35, 48, 30, 80, 88,  4, 23, 43, 87, 73, 58, 39, 54, 44, 61,
# #  83, 24,  1, 52, 11, 63, 57, 17, 64, 67, 46, 66, 14, 26, 71, 21, 77, 16, 62,
# #  18, 37,  7, 13,  8, 22, 55, 56, 76,  2, 15,  9, 84, 78, 65, 12, 50, 29, 33,
# #  32, 47, 19, 53,  0, 38, 70, 69, 74, 10, 82, 59,  5, 81, 51, 86, 27, 89, 45,
# #  42, 31,  3, 68, 28, 36, 75, 72, 40,  6, 34, 41, 79]
# # 得到kshell方法的节点重要性排序
# kshelldata = pd.read_csv('result-kshell2从大到小.csv')
# y = kshelldata['nodeid'].array
# print(y)
# # [54, 25, 71, 43, 46,  8, 58,  1, 73, 30, 17, 60, 49, 52, 57, 66, 61, 83, 63,
# #  21, 35,  4, 18, 85, 48,  7, 80, 39, 88, 11, 26, 24, 87, 62, 23, 44, 64, 55,
# #  37, 13, 50, 84, 78, 65, 14, 12,  2, 56, 22, 67, 16, 76,  9, 77, 29, 15, 32,
# #  33, 47, 19, 53,  5, 70,  0, 10, 38, 82, 59, 81, 74, 69, 51, 86, 89, 27, 42,
# #  45, 31,  3, 68, 36, 28, 75, 72, 40, 34, 79,  6, 41]
