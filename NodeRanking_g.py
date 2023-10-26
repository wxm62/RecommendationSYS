import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np
import time


# 生成拓扑图，三个参数
def get_Graph3(filename):
    file_name = filename
    g = nx.Graph()
    with open(file_name) as file:
        for line in file:
            head, tail, time = [str(x) for x in line.split()]  # 如果节点使用数组表示的可以将str(x)改为int( x)
            g.add_edge(head, tail)
        # nx.draw(g, node_color='yellow', with_labels=True, node_size=800, alpha=0.95, font_size=10)
        # plt.show()
        # print(head, tail)
    return g


# 生成拓扑图，两个参数
def get_Graph2(filename):
    file_name = filename
    g = nx.Graph()
    with open(file_name) as file:
        for line in file:
            head, tail = [str(x) for x in line.split()]  # 如果节点使用数组表示的可以将str(x)改为int( x)
            g.add_edge(head, tail)
        # nx.draw(g, node_color='yellow', with_labels=True, node_size=800, alpha=0.95, font_size=10)
        # plt.show()
        # print(head, tail)
    return g


# 输出绘制拓扑图
def print_graph(G):
    plt.figure()  # 创建一幅图
    # 输出网络拓扑图
    nx.draw(G, node_color='yellow', with_labels=True, node_size=800, alpha=0.95, font_size=10)
    # print(G.nodes())
    plt.show()


# 得到某个节点的度值
def gDegree(G, nodeid):
    """
    将G.degree()的返回值变为字典
    """
    node_degrees_dict = {}
    for i in G.degree():
        node_degrees_dict[i[0]] = i[1]
    default_info = '该节点编号不存在'
    result = node_degrees_dict.get(nodeid, default_info)
    return result


# 一个节点的节点度越大就意味着这个节点的度中心性越高，该节点在网络中就越重要。
def centrality(G):
    # 计算度中心性,降序
    dc = nx.algorithms.centrality.degree_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 以经过某个节点的最短路径数目来刻画节点重要性的指标
def betweenness(G, id):
    # 计算介数中心性,降序
    dc = nx.betweenness_centrality(G)
    # print(dc[id])
    return dc[id]
    # return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 反映在网络中某一节点与其他节点之间的接近程度。将一个节点到所有其他节点的最短路径距离的累加起来的倒数表示接近性中心性。即对于一个节点，它距离其他节点越近，那么它的接近性中心性越大。
def closeness(G):
    # 计算接近中心性,降序
    dc = nx.closeness_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 计算cita的值,计算i的邻居节点和i的BC值得和
def bizhi(G):
    df = pd.read_csv('result_bc.csv')
    df2 = pd.read_csv('result_cc.csv')
    DF = pd.DataFrame()
    tar = []
    all_bc = []
    sum_bc = 0

    for row in df.itertuples():
        # 获取第一列的数值
        tar = getattr(row, 'node')
        print(tar)
        # 计算每个节点的邻居
        nodes = list(nx.neighbors(G, str(tar)))
        for i in nodes:
            sum_bc = sum_bc + betweenness(G, i)
        all_bc.append(sum_bc)
        bizhi_label = pd.Series([int(tar), sum_bc])
        bizhi_label.index = ('node', 'neighbour_bc')
        DF = DF.append(bizhi_label, ignore_index=True)
        sum_bc = 0
    DF.to_csv("result_neighbours_bc.csv", index=False)


def imp_node_g():
    df1 = pd.read_csv('result_sort_kshell.csv')
    df2 = pd.read_csv('result_neighbours_bc.csv')
    df3 = pd.read_csv('result_salton.csv')
    df4 = pd.read_csv('result_bc.csv')
    df5 = pd.read_csv('result_cc.csv')
    all_salton = []
    DF = pd.DataFrame()
    DF2 = pd.DataFrame()
    DF['nodeid'] = df1['nodeid']

    for i in range(0, 1005):
        tar = df3[df3['node1'] == i]['result']
        sum = tar.sum()
        all_salton.append(sum)
        sum = 0
    print(all_salton)
    all_salton.remove(0)

    for k in range(0, 987):
        salton_label = pd.Series([k, all_salton[k]])
        salton_label.index = ('nodeid', 'nodesalton')
        DF2 = DF2.append(salton_label, ignore_index=True)
    DF2['nodeid'] = df1['nodeid']
    DF2.to_csv("result_nodesalton2.csv", index=False)

    for j in range(0, 988):
        DF['importance_g'] = df1['kshell'] + (df2['neighbour_bc'] * df4['BC'] * all_salton[j]) / (df5['CC'] * df5['CC'])
    DF.to_csv("result_imp_node_g.csv", index=False)


def imp_node_sort_g():
    df = pd.read_csv('result_imp_node_g.csv')
    df = df.sort_values(by=['importance_g'], ascending=False)
    DF = pd.DataFrame()
    DF['nodeid'] = df['nodeid']
    DF['importance_sort_g'] = df['importance_g']
    DF['importance_sort_chuli'] = round(df['importance_g'], 6)
    DF.to_csv("result_imp_node_sort_g.csv", index=False)
    # di = pd.DatetimeIndex(_dates,dtype='datetime64[ns]', freq=None)
    # pd.DataFrame({'nodeid': DF['nodeid'], 'importance': DF['importance_sort']},index=di).plot.line()


# 分辨率指标
def Monotonicity():
    df = pd.read_csv('result_imp_node_sort_g.csv')
    output = []
    similar = 0
    for i in range(1, 9):
        li = [i]
        tar = df[df['importance_sort_chuli'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 8):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (988 * 987))), 2)
    print(result)  # 0.9799569807171415   0.9798595289219623


def Monotonicity_kshell():
    df = pd.read_csv('result_sort_kshell.csv')
    output = []
    similar = 0
    for i in range(0, 35):
        li = [i]
        tar = df[df['kshell_guiyihua'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 34):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (988 * 987))), 2)
    print(result)  # 0.9692865257554854


def G_degree():
    G = get_Graph3('email-Eu-core-temporal.txt')
    DF = pd.DataFrame()
    degrees = []
    for i in range(0, 1005):
        degrees.append(gDegree(G, str(i)))
        salton_label = pd.Series([i, gDegree(G, str(i))])
        salton_label.index = ('nodeid', 'degree')
        DF = DF.append(salton_label, ignore_index=True)
    DF.to_csv('result_degree.csv', index=False)

    # degrees.remove('该节点编号不存在') 手动删除
    print(degrees)


def Monotonicity_degree():
    df = pd.read_csv('result_degree.csv')
    output = []
    similar = 0
    for i in range(0, 346):
        li = [i]
        tar = df[df['degree'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    output.remove(0)
    print('----')
    for j in range(len(output)):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (988 * 987))), 2)
    print(result)  # 0.9572773349558502


def Monotonicity_bc():
    df = pd.read_csv('result_bc.csv')
    output = []
    similar = 0
    for i in [0, 4.233672101048874e-05]:
        li = [i]
        tar = df[df['BC'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 2):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (988 * 987))), 2)
    print(result)  # 0.9552797402333552


def Monotonicity_cc():
    df = pd.read_csv('result_cc.csv')
    output = []
    similar = 0
    result = df.groupby(['CC']).count()
    # print(result)
    result.to_csv('cc_group.csv', index=False)  # 将nr保存在文件中
    # 手动删除1
    for i in range(1, 6):
        li = [i]
        tar = result[result['node'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 5):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (988 * 987))), 2)
    print(result)  # 0.5840389194388277

def attribute(G):
    df = pd.read_csv('result_degree.csv')
    result = 0
    for row in df.itertuples():
        # 获取第一列的数值
        tar = getattr(row, 'nodeid')
        result = result + nx.clustering(G, str(tar))
    print("email网络的节点聚类系数和为：" + str(result))
    avg_result = result / 986
    print("email网络的平均聚类系数和为：" + str(avg_result))
    print(round(avg_result, 4))
    r = nx.degree_pearson_correlation_coefficient(G, weight=None, nodes=None)
    print("email网络的同配系数为：" + str(r))
    print(round(r, 4))
    L = nx.average_shortest_path_length(G, weight=None, method=None)
    print("email网络的同配系数为：" + str(L))
    print(round(L, 4))
def diameter(G):
    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping)
    maxh1 = max(H) + 1
    A = np.zeros((maxh1, maxh1))
    for i in range(maxh1):
        for j in range(maxh1):
            if nx.has_path(H, i, j) == True:
                A[i][j] = nx.shortest_path_length(H, i, j)
            else:
                A[i][j] = 0
    print('图的直径：', A.max())

if __name__ == '__main__':

    G = get_Graph3('email-Eu-core-temporal.txt')
    # Monotonicity()
    imp_node_g()
