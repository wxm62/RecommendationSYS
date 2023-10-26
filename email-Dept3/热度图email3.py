import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, 0.10214504596527069, -0.0020429009193054137, -0.026557711950970377, -0.024514811031664963],
             [0.10214504596527069, 1.0, 0.1634320735444331, 0.26251276813074564, 0.15321756894790603],
             [-0.0020429009193054137, 0.1634320735444331, 1.0, 0.008171603677221655, 0.1041879468845761],
             [-0.026557711950970377,0.26251276813074564, 0.008171603677221655, 1.0, 0.0602655771195097],
             [-0.024514811031664963, 0.15321756894790603, 0.1041879468845761, 0.0602655771195097, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.3,
                       vmin=-0.03,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Email-dept3)', fontsize=18)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图email-dept3.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
