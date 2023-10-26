import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, 0.15067862893949852, 0.051913196840733075, 0.06464228203358638, 0.17322291235334714],
             [0.15067862893949852, 1.0, 0.025534851621808144, 0.11586534774940573, 0.14193696802392455],
             [0.051913196840733075, 0.025534851621808144, 1.0, -0.001610305958132045, 0.1066635994172226],
             [0.06464228203358638,0.11586534774940573, -0.001610305958132045, 1.0, -0.01403266620657925],
             [0.17322291235334714, 0.14193696802392455, 0.1066635994172226,-0.01403266620657925, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.2,
                       vmin=-0.02,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Email-dept2)', fontsize=18)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图email-dept2.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
