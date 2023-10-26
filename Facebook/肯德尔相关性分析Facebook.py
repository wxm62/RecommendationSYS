# Tau_b
from scipy.stats.stats import kendalltau
import numpy as np
import pandas as pd

# data = pd.read_csv('importance_facebook从大到小.csv')
# dat1 = data['node'].array

# kshelldata = pd.read_csv('result-kshell2从大到小.csv')
# dat1 = kshelldata['nodeid'].array #-0.01295271174789247

# bcdata = pd.read_csv('result_bc从大到小.csv')
# dat1 = bcdata['node'].array #-0.0065125366330185605

ccdata = pd.read_csv('result_cc从大到小.csv')
dat1 = ccdata['node'].array  # 0.004992944751980897

degreedata = pd.read_csv('result_degree从大到小.csv')
dat2 = degreedata['nodeid'].array  # 0.10673323926335974

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
