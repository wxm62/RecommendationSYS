import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def zhenghe_data():
    DF = pd.DataFrame()
    df1 = pd.read_csv('result_sort_kshell.csv')
    df2 = pd.read_csv('result_neighbours_bc.csv')
    df3 = pd.read_csv('result_bc.csv')
    df4 = pd.read_csv('result_cc.csv')
    df5 = pd.read_csv('result_nodesalton.csv')
    DF['node'] = df1['nodeid']
    DF['kshell'] = round(df1['kshell_guiyihua'], 6)
    DF['quanzhong'] = round((df2['neighbour_bc'] * df3['BC']) / (df4['CC'] * df4['CC']), 6)
    DF['salton'] = round(df5['nodesalton'], 6)
    DF.to_csv("result_yanjiu1.csv", index=False)


# 先删除 𝐾𝑠 = 0且Salton指标为0的节点，再删除权重为零的节点
def shuju_quzao():
    df = pd.read_csv('result_yanjiu1.csv')
    qu_kshell = df[(df['kshell'] != 0.0)]  # 891
    qu_salton = qu_kshell[(qu_kshell['salton'] != 0.0)]  # 874
    qu_quanzhong = qu_salton[(qu_salton['quanzhong'] != 0.0)]  # 768
    # new_df = df[df.loc['kshell'] != 0.0].dropna() & df[df.loc['salton'] != 0.0].dropna()

    print(qu_quanzhong)
    qu_quanzhong.to_csv('result_yanjiuquzao.csv', index=False)


def core_fringe_node():
    df = pd.read_csv('result_yanjiuquzao.csv')
    for i in range(0, 768):
        df['import'] = round(df['kshell'] + df['quanzhong'] * df['salton'],6)

    df.to_csv('result_yanjiuquzao.csv', index=False)
    print("计算节点重要性结束")
    df = df.sort_values(by=['import'], ascending=False)
    df2 = df.head(154)
    df2.to_csv('corenodes.csv', index=False)
    print("获得核心节点")
    df3 = df.tail(614)
    df3.to_csv('fringenodes.csv', index=False)


if __name__ == '__main__':
    core_fringe_node()
