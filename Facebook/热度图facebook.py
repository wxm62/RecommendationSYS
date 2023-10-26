import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def dataPlot():
    '''
    基于相关性系数计算结果来绘制
    '''
    data1 = [[1.0, -0.05510329606715149, -0.002930641484858352, -0.025652158182278666, -0.01295271174789247],
             [-0.05510329606715149, 1.0, -0.017475306631933137,0.03954556966605159, 0.10673323926335974],
             [-0.002930641484858352,-0.017475306631933137, 1.0, 0.08585694127862803, -0.0065125366330185605],
             [-0.025652158182278666,0.03954556966605159,0.08585694127862803, 1.0, 0.004992944751980897],
             [-0.01295271174789247, 0.10673323926335974, -0.0065125366330185605, 0.004992944751980897, 1.0],
             ]

    data1 = np.array(data1)
    fig, ax = plt.subplots(figsize=(10, 10))
    key_list = ['Kshell', 'DC', 'BC', 'CC', 'KSCNR']
    cmap = sns.heatmap(pd.DataFrame(np.round(data1, 4), columns=key_list, index=key_list), annot=True, vmax=0.2,
                       vmin=-0.02,
                       xticklabels=True,
                       yticklabels=True, square=True, cmap="YlGnBu")  # YlOrBr
    ax.set_title('Heat Map(Facebook)', fontsize=18)
    # ax.set_ylabel('Y', fontsize=18)
    # ax.set_xlabel('X', fontsize=18)

    plt.savefig('热度图Facebook.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # randomPlot()
    dataPlot()
