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
    df = pd.read_csv('importance_dept3.csv')
    # 重复值只有4个，均为0
    similar = 4 * 3
    print(similar)
    result = pow((1 - (similar / (89 * 88))), 2)
    print(result)  # 0.9969379961833853


def Monotonicity_kshell():
    df = pd.read_csv('result_sort_kshell.csv')
    output = []
    similar = 0
    for i in range(1, 18):
        li = [i]
        tar = df[df['kshell'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 17):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (89 * 88))), 2)
    print(result)  # 0.6145983425166495


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
    G = get_Graph3('email-Eu-core-temporal-Dept3.txt')
    DF = pd.DataFrame()
    degrees = []
    for i in range(0, 90):
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
    for i in range(1, 60):
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
    result = pow((1 - (similar / (89 * 88))), 2)
    print(result)  # 0.9610607877793208


def Monotonicity_bc():
    df = pd.read_csv('result_bc.csv')
    # 重复值只有5个，均为0
    similar = 5 * 4
    print(similar)
    result = pow((1 - (similar / (89 * 88))), 2)
    print(result)  # 0.994899268708246


def Monotonicity_cc():
    df = pd.read_csv('result_cc.csv')
    output = []
    similar = 0
    result = df.groupby(['CC']).count()
    # print(result)
    result.to_csv('cc_group.csv', index=False)  # 将nr保存在文件中
    # 手动删除1
    for i in range(2, 5):
        li = [i]
        tar = result[result['node'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 3):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (89 * 88))), 2)
    print(result)  # 0.9268241341929236


if __name__ == '__main__':
    G = get_Graph3('email-Eu-core-temporal-Dept3.txt')
    # Monotonicity()
    # Monotonicity_kshell()
    #
    # G_degree()

    # Monotonicity_degree()
    # Monotonicity_bc()
    Monotonicity_cc()
