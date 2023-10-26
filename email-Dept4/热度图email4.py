import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, 0.028668464688842273, -0.004894615922485266, -0.011687144141444412, 0.03546099290780142],
             [0.028668464688842273, 1.0, 0.0026970332634102486, -0.05683747877334932, 0.270002996703626],
             [-0.004894615922485266, 0.0026970332634102486, 1.0, -0.010088902207571672, 0.10658275896513834],
             [-0.011687144141444412,-0.05683747877334932, -0.010088902207571672, 1.0, 0.03745879532514235],
             [0.03546099290780142, 0.270002996703626, 0.10658275896513834, 0.03745879532514235, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.3,
                       vmin=-0.05,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Email-dept4)', fontsize=18)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图email-dept4.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
