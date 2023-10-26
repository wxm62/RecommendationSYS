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
        tar = df[df['kshell'].isin(li)]
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
    print(result)  # 0.934093955621396


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


# 分辨率指标
def Monotonicity_quzao():
    df = pd.read_csv('result_yanjiuquzao.csv')
    output = []
    similar = 0
    li = [0.151536]
    similar = 2 * (2 - 1)
    print(similar)
    result = pow((1 - (similar / (768 * 767))), 2)
    print(result)  # 0.9999932094856695


if __name__ == '__main__':
    G = get_Graph3('email-Eu-core-temporal.txt')
    # Monotonicity_quzao()
    Monotonicity_kshell()
