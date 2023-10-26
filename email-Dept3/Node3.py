import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np
import time
import psutil


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


# K壳分解，按照ks值
def kshell(G):
    ks_results = {}
    DF = pd.DataFrame()
    # print(G)
    for num in range(1, 18):
        ks = nx.k_shell(G, num)  # 第二层节点
        # if not ks:
        #     break

        # ks_results.update(key=num, value=ks.nodes())
        ks_results[num] = ks.nodes()
        print(ks.nodes())
    for i in range(1, 18):
        print("ks={}:{}".format(i, ks_results[i]))
        k_label = pd.Series([i, ks_results[i]])
        k_label.index = ('kshell', 'result')
        DF = DF.append(k_label, ignore_index=True)
    # print("ks=1:{}".format(ks_results[1]))
    DF.to_csv("result-kshell.csv", index=False)


# K壳分解，按照节点
def kshell2(G):
    ks_results = {}
    DF = pd.DataFrame()

    # print(G)
    for num in range(1, 18):
        ks = nx.k_shell(G, num)  # 第num层的子图
        # if not ks:
        #     break
        # ks_results.update(key=num, value=ks.nodes())
        ks_results[num] = ks.nodes()
    for i in range(1, 18):
        for value in ks_results[i]:
            k_label = pd.Series([value, i])
            k_label.index = ('nodeid', 'kshell')
            DF = DF.append(k_label, ignore_index=True)
    DF.to_csv("result-kshell2.csv", index=False)


# 排序和归一化
def sort_kshell():
    df1 = pd.read_csv('result-kshell2.csv')
    df1 = df1.sort_values(by=['nodeid'], ascending=True)
    # df2 = pd.read_csv('result_bc.csv')
    DF1 = pd.DataFrame()
    DF1['nodeid'] = df1['nodeid']
    DF1['kshell'] = df1['kshell']
    # 保留六位小数
    # DF1['kshell_guiyihua'] = round(((df1['kshell'] - 1) / (34 - 1)), 6)
    # 不保留六位小数
    DF1['kshell_guiyihua'] = ((df1['kshell'] - 1) / (17 - 1))
    DF1.to_csv("result_sort_kshell.csv", index=False)


# 计算Salton指标
def get_Salton(G):
    DF = pd.DataFrame()
    for i in G.nodes():
        for j in G.nodes():
            # 源节点到源节点没有度,两个节点之间没有共同邻居，停止
            if i != j:
                if gDegree(G, str(i)) >= 1 and gDegree(G, str(i)) >= 1:
                    if nx.common_neighbors(G, str(i), str(j)):
                        if len(list(nx.common_neighbors(G, str(i), str(j)))) != 0:
                            cneighbor = len(list(nx.common_neighbors(G, str(i), str(j))))
                            degreesum = pow((gDegree(G, str(i)) * gDegree(G, str(j))), 0.5)
                            result = cneighbor / degreesum
                            salton_label = pd.Series([i, j, result])
                            salton_label.index = ('node1', 'node2', 'result')
                            DF = DF.append(salton_label, ignore_index=True)
                            # salton.append(result)
                            # print(salton)
                            print("节点{}和节点{}的salton指标值：{},共同邻居个数为：{},度之和的根号为：{}".format(i, j, result, cneighbor,
                                                                                        degreesum))
                            print("DF:", DF)
                else:
                    break
    DF.to_csv("result_salton.csv", index=False)


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


def bc_cc(G):
    DF1 = pd.DataFrame()
    DF2 = pd.DataFrame()
    DF3 = pd.DataFrame()
    BC = nx.betweenness_centrality(G)  # [('978', 0.07738095238095238), ('4', 0.023384353741496597)]
    BC = sorted(BC.items(), key=lambda x: int(x[0]), reverse=False)  # 计算接近中心性,降序
    print(BC)
    for bc in BC:
        bc_label = pd.Series([bc[0], bc[1]])
        bc_label.index = ('node', 'BC')
        DF1 = DF1.append(bc_label, ignore_index=True)
    DF1.to_csv("result_bc.csv", index=False)

    CC = nx.closeness_centrality(G)
    CC = sorted(CC.items(), key=lambda x: int(x[0]), reverse=False)
    for cc in CC:
        cc_label = pd.Series([cc[0], cc[1]])
        cc_label.index = ('node', 'CC')
        DF2 = DF2.append(cc_label, ignore_index=True)
    DF2.to_csv("result_cc.csv", index=False)


def neighbours_bc(G):
    df = pd.read_csv('result_bc.csv')
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
    print('neighbours_bc保存成功')


def importance_node():
    df1 = pd.read_csv('result_sort_kshell.csv')
    df2 = pd.read_csv('result_neighbours_bc.csv')
    df3 = pd.read_csv('result_salton.csv')
    df4 = pd.read_csv('result_bc.csv')
    df5 = pd.read_csv('result_cc.csv')
    all_salton = []
    DF = pd.DataFrame()
    DF2 = pd.DataFrame()
    DF['nodeid'] = df1['nodeid']

    for row in df1.itertuples():
        # 获取第一列的数值
        tar = getattr(row, 'nodeid')
        i = df3[df3['node1'] == tar]['result']
        sum = i.sum()
        all_salton.append(sum)
        sum = 0
    print(all_salton)
    print(len(all_salton))

    for k in range(0, 89):
        salton_label = pd.Series([int(k), all_salton[k]])
        salton_label.index = ('nodeid', 'nodesalton')
        DF2 = DF2.append(salton_label, ignore_index=True)
    DF2['nodeid'] = df1['nodeid']
    DF2.to_csv("result_nodesalton.csv", index=False)
    print('计算每个节点salton指标结束')


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
    DF['import'] = round(DF['kshell'] + DF['quanzhong'] * DF['salton'], 6)
    DF.to_csv("importance_dept3.csv", index=False)


# 先删除 𝐾𝑠 = 0且Salton指标为0的节点，再删除权重为零的节点
def shuju_quzao():
    df = pd.read_csv('importance_dept3.csv')
    qu_kshell = df[(df['kshell'] != 0.0)]  # 86
    qu_salton = qu_kshell[(qu_kshell['salton'] != 0.0)]  # 86
    qu_quanzhong = qu_salton[(qu_salton['quanzhong'] != 0.0)]  # 85
    # new_df = df[df.loc['kshell'] != 0.0].dropna() & df[df.loc['salton'] != 0.0].dropna()

    print(qu_quanzhong)
    qu_quanzhong.to_csv('result_yanjiuquzao.csv', index=False)

# 数据去噪的核心节点
# def core_fringe_node():
#     df = pd.read_csv('result_yanjiuquzao.csv')
#     for i in range(0, 90):
#         df['import'] = round(df['kshell'] + df['quanzhong'] * df['salton'], 6)
#
#     df.to_csv('result_yanjiuquzao.csv', index=False)
#     print("计算节点重要性结束")
#     df = df.sort_values(by=['import'], ascending=False)
#     df2 = df.head(17)  # 前20% 17 67
#     df2.to_csv('corenodes.csv', index=False)
#     print("获得核心节点")
#     df3 = df.tail(67)
#     df3.to_csv('fringenodes.csv', index=False)


def core_fringe_node():
    df = pd.read_csv('importance_dept3重要性从大到小.csv')
    df2 = df.head(41)  # 前20% 17 67
    df2.to_csv('corenodes3.csv', index=False)
    print("获得核心节点")
    df3 = df.tail(48)
    df3.to_csv('fringenodes3.csv', index=False)

# 分辨率指标,无重复值
def Monotonicity():
    df = pd.read_csv('result_yanjiuquzao.csv')
    output = []
    similar = 0
    for i in ():
        li = [i]
        tar = df[df['import'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 2):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (112 * 111))), 2)
    print(result)  # 1


def attribute(G):
    df = pd.read_csv('result_degree.csv')
    result = 0
    for row in df.itertuples():
        # 获取第一列的数值
        tar = getattr(row, 'nodeid')
        result = result + nx.clustering(G, str(tar))
    print("email-dept3网络的节点聚类系数和为：" + str(result))
    avg_result = result / 89
    print("email-dept3网络的平均聚类系数和为：" + str(avg_result))
    print(round(avg_result, 4))
    r = nx.degree_pearson_correlation_coefficient(G, weight=None, nodes=None)
    print("email-dept3网络的同配系数为：" + str(r))
    print(round(r, 4))


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
    G = get_Graph3('email-Eu-core-temporal-Dept3.txt')
    # print_graph(G)

    start_cpu_time = time.process_time()
    start_time = time.time()

    kshell(G)
    kshell2(G)
    sort_kshell()
    get_Salton(G)
    bc_cc(G)
    neighbours_bc(G)

    importance_node()
    zhenghe_data()
    shuju_quzao()
    core_fringe_node()

    end_cpu_time = time.process_time()
    cpu_time = end_cpu_time - start_cpu_time
    print("CPU时间:", cpu_time)

    end_time = time.time()
    execution_time = end_time - start_time
    print("执行时间:", execution_time)

    cpu_load = psutil.cpu_percent(interval=1)
    print("CPU负载:", cpu_load)
    # Monotonicity()
    # diameter(G)
    # core_fringe_node()
