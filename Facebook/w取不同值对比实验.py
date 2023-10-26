import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np


def comparation():
    df = pd.read_csv('result_yanjiuquzao.csv')
    df2 = pd.read_csv('result-kshell2.csv')
    DF = pd.DataFrame()
    DF['nodeid'] = df['node']
    DF['importance_w0'] = 2 * df['quanzhong'] * df['salton']
    DF = DF.sort_values(by=['importance_w0'], ascending=False)
    DF.to_csv("result_imp_w0.csv", index=False)

    DF1 = pd.DataFrame()
    DF1['nodeid'] = df['node']
    DF1['importance_w0.5'] = 0.5 * df['kshell'] + 1.5 * df['quanzhong'] * df['salton']
    DF1 = DF1.sort_values(by=['importance_w0.5'], ascending=False)
    DF1.to_csv("result_imp_w0.5.csv", index=False)

    DF2 = pd.DataFrame()
    DF2['nodeid'] = df['node']
    DF2['importance_w1.5'] = 1.5 * df['kshell'] + 0.5 * df['quanzhong'] * df['salton']
    DF2 = DF2.sort_values(by=['importance_w1.5'], ascending=False)
    DF2.to_csv("result_imp_w1.5.csv", index=False)

    DF3 = pd.DataFrame()
    DF3['nodeid'] = df['node']
    DF3['importance_w2'] = 2 * df['kshell']
    DF3 = DF3.sort_values(by=['importance_w2'], ascending=False)
    DF3.to_csv("result_imp_w2.csv", index=False)

    DF4 = pd.DataFrame()
    DF4['nodeid'] = df['node']
    DF4['importance_w1'] = df['kshell'] + df['quanzhong'] * df['salton']
    DF4 = DF4.sort_values(by=['importance_w1'], ascending=False)
    DF4.to_csv("result_imp_w1.csv", index=False)


if __name__ == '__main__':
    comparation()
