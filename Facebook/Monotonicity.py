import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


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


# 分辨率指标
def Monotonicity():
    df = pd.read_csv('importance_facebook.csv')
    output = []
    similar = 0
    for i in (0, 0.052632, 0.062554, 0.105263, 0.157895, 0.157948, 0.191344, 0.210526, 0.210607, 0.210631, 0.368421,
              0.421053, 0.51416, 0.614537, 0.974819):
        li = [i]
        tar = df[df['import'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 15):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (333 * 332))), 2)
    print(result)  # 0.9808802920059408


def Monotonicity_kshell():
    df = pd.read_csv('result_sort_kshell.csv')
    output = []
    similar = 0
    for i in range(0, 21):
        li = [i]
        tar = df[df['kshell'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 21):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (333 * 332))), 2)
    print(result)  # 0.865184686398882


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


def G_degree():
    G = get_Graph2('Facebook-0节点数据集.txt')
    DF = pd.DataFrame()
    degrees = []
    for i in range(0, 348):
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
    # 最大度值+1
    for i in range(1, 78):
        li = [i]
        tar = df[df['degree'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    # output.remove(0)
    print('----')
    for j in range(len(output)):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (333 * 332))), 2)
    print(result)  # 0.9241794469595582


def Monotonicity_bc():
    df = pd.read_csv('result_bc.csv')
    output = []
    similar = 0
    for i in (0, 3.63993739307683E-06, 0.0000066732185539742, 9.09984348269209E-06, 0.00292104975794416,
              0.00580570014195755, 0.00586029920285371, 0.011465802788192, 0.0172927358982758, 0.0204440725732606):
        li = [i]
        tar = df[df['BC'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    output = [60, 2, 2, 4, 2, 3, 4, 3, 2, 3]
    print(output)
    for j in range(0, 10):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (333 * 332))), 2)
    print(result)  # 0.9361099832686604


def Monotonicity_cc():
    df = pd.read_csv('result_cc.csv')
    output = []
    similar = 0
    result = df.groupby(['CC']).count()
    # print(result)
    result.to_csv('cc_group.csv', index=False)  # 将nr保存在文件中
    # 手动删除1
    for i in range(2, 7):
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
    result = pow((1 - (similar / (333 * 332))), 2)
    print(result)  # 0.9475919027835583


if __name__ == '__main__':
    G = get_Graph2('Facebook-0节点数据集.txt')
    # Monotonicity()
    # Monotonicity_kshell()

    # G_degree()

    # Monotonicity_degree()
    # Monotonicity_bc()
    Monotonicity_cc()
    # Monotonicity_quzao()
