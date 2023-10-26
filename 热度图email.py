import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


#
# def randomPlot():
#     '''
#     构造随机数矩阵来绘制热力图
#     '''
#     data = np.random.rand(8, 8)
#     print(data)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     key_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
#     sns.heatmap(pd.DataFrame(np.round(data, 4), columns=key_list, index=key_list), annot=True, vmax=1, vmin=0,
#                 xticklabels=True,
#                 yticklabels=True, square=True, cmap="YlGnBu")
#     ax.set_title(' Heat Map ', fontsize=18)
#     ax.set_ylabel('Y', fontsize=18)
#     ax.set_xlabel('X', fontsize=18)
#     plt.savefig('Random.png')


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, 0.02027779779862234, -0.004956703493580173, -0.020257204929932764, -0.02146394703514173],
             [0.02027779779862234, 1.0, 0.03336662513771481, 0.025298339185140187, 0.02127449264319766],
             [-0.004956703493580173, 0.03336662513771481, 1.0, 0.014272917288742908, -0.03643084399872324],
             [-0.020257204929932764, 0.025298339185140187, 0.014272917288742908, 1.0, 0.02260891053428198],
             [-0.02146394703514173, 0.02127449264319766, -0.03643084399872324, 0.02260891053428198, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.1,
                       vmin=-0.05,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Email)', fontsize=18)
    ax.tick_params(labelsize=5)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图emailall.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
