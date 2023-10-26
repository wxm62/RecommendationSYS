import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np


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


# 计算图中所有节点度的之和
def sumD(G):
    """
    计算G中度的和
    """
    G_degrees = gDegree(G)
    sum = 0
    for v in G_degrees.values():
        sum += v
    return sum


# K壳分解，按照ks值
def kshell(G):
    ks_results = {}
    DF = pd.DataFrame()
    # print(G)
    for num in range(1, 35):
        a = nx.k
        ks = nx.k_shell(G, num)  # 第二层节点
        if not ks:
            break
        # ks_results.update(key=num, value=ks.nodes())
        ks_results[num] = ks.nodes()
        # print(ks.nodes())
    for i in range(1, 35):
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
    for num in range(1, 35):
        ks = nx.k_shell(G, num)  # 第num层的子图
        if not ks:
            break
        # ks_results.update(key=num, value=ks.nodes())
        ks_results[num] = ks.nodes()
    for i in range(1, 35):
        for value in ks_results[i]:
            k_label = pd.Series([value, i])
            k_label.index = ('nodeid', 'kshell')
            DF = DF.append(k_label, ignore_index=True)
    DF.to_csv("result-kshell2.csv", index=False)


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


def shangquan():
    '''1.输入数据'''
    print("请输入参评数目：")
    n = eval(input())
    print("请输入指标数目：")
    m = eval(input())
    print("请输入矩阵：")
    X = np.zeros(shape=(n, m))
    for i in range(n):
        X[i] = input().split(" ")
        X[i] = list(map(float, X[i]))
    print("输入矩阵为：\n{}".format(X))

    '''2.归一化处理'''
    X = X.astype('float')
    for j in range(m):
        X[:, j] = X[:, j] / sum(X[:, j])
    print("归一化矩阵为：\n{}".format(X))

    '''3.计算概率矩阵p'''
    p = X
    for j in range(m):
        p[:, j] = X[:, j] / sum(X[:, j])

    '''4.计算熵值'''
    E = np.array(X[0, :])
    for j in range(m):
        E[j] = -1 / np.log(n) * sum(p[:, j] * np.log(p[:, j] + 1e-5))
    print("熵值矩阵为：\n{}".format(E))

    '''5.计算熵权'''
    w = (1 - E) / sum(1 - E)
    print("熵权矩阵为：\n{}".format(w))

    '''6.加权后的数据'''
    R = X * w
    print("加权后数据矩阵为：\n{}".format(R))


# 一个节点的节点度越大就意味着这个节点的度中心性越高，该节点在网络中就越重要。
def centrality(G):
    # 计算度中心性,降序
    dc = nx.algorithms.centrality.degree_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 以经过某个节点的最短路径数目来刻画节点重要性的指标
def betweenness(G):
    # 计算介数中心性,降序
    dc = nx.betweenness_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 反映在网络中某一节点与其他节点之间的接近程度。将一个节点到所有其他节点的最短路径距离的累加起来的倒数表示接近性中心性。即对于一个节点，它距离其他节点越近，那么它的接近性中心性越大。
def closeness(G):
    # 计算接近中心性,降序
    dc = nx.closeness_centrality(G)
    return sorted(dc.items(), key=lambda x: x[1], reverse=True)


# 计算cita的值
def bizhi(G):
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

    DF3['nodeid'] = DF1['node']
    # 小数点保留6位
    # DF3['BC'] = round(DF1['BC'], 6)
    # DF3['CC'] = round(DF2['CC'], 6)
    # DF3['bizhi'] = round(DF3['BC'] / DF3['CC'], 6)
    # 不保留小数
    DF3['BC'] = DF1['BC']
    DF3['CC'] = DF2['CC']
    DF3['bizhi'] = DF3['BC'] / DF3['CC']

    DF3.to_csv("result_bizhi.csv", index=False)

    print(DF1['node'])


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
    DF1['kshell_guiyihua'] = ((df1['kshell'] - 1) / (34 - 1))
    DF1.to_csv("result_sort_kshell.csv", index=False)


def imp_node():
    df1 = pd.read_csv('result_sort_kshell.csv')
    df2 = pd.read_csv('result_bizhi.csv')
    df3 = pd.read_csv('result_salton.csv')
    all_salton = []
    DF = pd.DataFrame()
    DF['nodeid'] = df1['nodeid']
    for i in range(0, 1004):
        tar = df3[df3['node1'] == i]['result']
        sum = tar.sum()
        all_salton.append(sum)
        sum = 0
    print(all_salton)
    all_salton.remove(0)
    for j in range(0, 987):
        DF['importance'] = df1['kshell'] + df2['bizhi'] * all_salton[j]
    DF.to_csv("result_imp_node.csv", index=False)


def imp_node_sort():
    df = pd.read_csv('result_imp_node.csv')
    df = df.sort_values(by=['importance'], ascending=False)
    DF = pd.DataFrame()
    DF['nodeid'] = df['nodeid']
    DF['importance_sort'] = df['importance']
    DF['importance_sort_chuli'] = round(df['importance'], 6)
    DF.to_csv("result_imp_node_sort.csv", index=False)
    # di = pd.DatetimeIndex(_dates,dtype='datetime64[ns]', freq=None)
    # pd.DataFrame({'nodeid': DF['nodeid'], 'importance': DF['importance_sort']},index=di).plot.line()


# 分辨率指标
def Monotonicity():
    df = pd.read_csv('result_imp_node_sort.csv')
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
    print(result)  # 0.9799569807171415


if __name__ == '__main__':
    G = get_Graph3('email-Eu-core-temporal.txt')
    # print_graph(G)
    # bizhi(G)
    # sort_kshell()
    # imp_node()
    # imp_node_sort()

    Monotonicity()

    # G = get_Graph3('shuju.txt')
    # kshell2(G)
    # aa = []
    # df = pd.read_csv('result-kshell.csv')
    # for i in range(0, 34):
    #     aa = df['result'].iloc[i:i + 1]
    #     for line in aa:
    #         value = [str(x) for x in line.split(',')]  # 如果节点使用数组表示的可以将str(x)改为int( x)
    #         print(line)

    # get_Salton(G)
    # kshell(G)
    # bizhi(G)

    # print(betweenness(G))  # [('978', 0.07738095238095238), ('4', 0.023384353741496597)]
    # print(closeness(G))

    # print(G.degree('494'))
    # print_graph(G)
    # kshell(G)
    # print(gDegree(G, '4'))


