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
    df = pd.read_csv('importance_dept4.csv')
    output = []
    similar = 0
    for i in (1.01341, 0.870267, 0.861451, 0.438169, 0.357143, 0.214286, 0.142857, 0.071429, 0):
        li = [i]
        tar = df[df['import'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 9):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (142 * 141))), 2)
    print(result)  # 0.9749861008449592


def Monotonicity_kshell():
    df = pd.read_csv('result_sort_kshell.csv')
    output = []
    similar = 0
    for i in range(0, 16):
        li = [i]
        tar = df[df['kshell'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 16):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (142 * 141))), 2)
    print(result)  # 0.7767435144086485


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
    G = get_Graph3('email-Eu-core-temporal-Dept4.txt')
    DF = pd.DataFrame()
    degrees = []
    for i in range(0, 146):
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
    for i in range(1, 30):
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
    result = pow((1 - (similar / (142 * 141))), 2)
    print(result)  # 0.9241794469595582


def Monotonicity_bc():
    df = pd.read_csv('result_bc.csv')
    output = []
    similar = 0
    for i in (0, 7.23693732812273E-06, 0.0000633232016210739, 0.000322856249907921, 0.00051952343700958,
              0.000834597688701032, 0.00146644473392193):
        li = [i]
        tar = df[df['BC'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    output = [38, 3, 2, 2, 2, 2, 2]
    print(output)
    for j in range(0, 7):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (142 * 141))), 2)
    print(result)  # 0.8630003549885955


def Monotonicity_cc():
    df = pd.read_csv('result_cc.csv')
    output = []
    similar = 0
    result = df.groupby(['CC']).count()
    # print(result)
    result.to_csv('cc_group.csv', index=False)  # 将nr保存在文件中
    # 手动删除1
    for i in range(2, 6):
        li = [i]
        tar = result[result['node'].isin(li)]
        sum = len(tar)
        output.append(sum)
    print(output)
    print('----')
    for j in range(0, 4):
        tar = output[j] * (output[j] - 1)
        similar = similar + tar
        # print(tar)
    print(similar)
    result = pow((1 - (similar / (142 * 141))), 2)
    print(result)  # 0.960051534960729


if __name__ == '__main__':
    G = get_Graph3('email-Eu-core-temporal-Dept4.txt')
    # Monotonicity()
    # Monotonicity_kshell()

    # G_degree()

    # Monotonicity_degree()
    # Monotonicity_bc()
    Monotonicity_cc()
    # Monotonicity_quzao()
