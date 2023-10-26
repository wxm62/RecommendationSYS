import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, 0.06691043584247468, 0.0034043626276635984, 0.04316395578531501, 0.05022485605009877],
             [0.06691043584247468, 1.0, 0.07649308620182406, 0.1356701550876308, 0.0733829277518598],
             [0.0034043626276635984, 0.07649308620182406, 1.0, 0.00710292943302652, 0.07123944017147901],
             [0.04316395578531501, 0.1356701550876308, 0.00710292943302652, 1.0, 0.06014373975539024],
             [0.05022485605009877, 0.0733829277518598, 0.07123944017147901,0.06014373975539024, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.15,
                       vmin=0,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Email-dept1)', fontsize=18)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图email-dept1.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
