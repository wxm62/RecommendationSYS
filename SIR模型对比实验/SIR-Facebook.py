import random
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySIR(y, t, lamda, mu, n):  # SIR 模型，导数函数
    i, s = y
    di_dt = lamda * s * i / n - mu * i  # di/dt = lamda*s*i-mu*i
    ds_dt = -lamda * s * i / n  # ds/dt = -lamda*s*i
    return np.array([di_dt, ds_dt])


number = 333  # 总人数
lamda = 1 / 15  # 感染率
mu = 0.1  # 恢复率
tEnd = 50  # 预测日期长度
t = np.arange(0, tEnd, 2)  # (start,stop,step)

i0 = 176 / number  # 患病者比例的初值
s0 = 1 - i0  # 易感者比例的初值
Y0 = (i0, s0)  # 微分方程组的初值

i1 = random.random()  # 患病者比例的初值
s1 = 1 - i1  # 易感者比例的初值
Y1 = (i1, s1)  # 微分方程组的初值
print(i1)

i2 = 333 / number  # 患病者比例的初值 到度值为55
s2 = 1 - i2  # 易感者比例的初值
Y2 = (i2, s2)  # 微分方程组的初值

ySIR0 = odeint(dySIR, Y0, t, args=(lamda, mu, number))  # SIR 模型
ySIR1 = odeint(dySIR, Y1, t, args=(lamda, mu, number))
ySIR2 = odeint(dySIR, Y2, t, args=(lamda, mu, number))

plt.title("SIR models in Facebook")
plt.xlabel('Infection Time in Facebook')
plt.ylabel('F(t)')
plt.axis([0, tEnd, -0.1, 1.1])

plt.plot(t, 1 - ySIR0[:, 0], 'red', marker='o', lw=1, label='KSCNR Initial Affect', linestyle='-',
         markerfacecolor='white')
plt.plot(t, 1 - ySIR1[:, 0], 'black', marker='^', lw=1, label='Random Initial Affect',
         markerfacecolor='white')
plt.plot(t, 1 - ySIR2[:, 0], 'blue', marker='*', lw=1, label='DC Initial Affect',
         markerfacecolor='white')
plt.legend(loc='lower right')
plt.tight_layout()  # 自动调整子图参数，使之自动填充整个图像区域
plt.savefig('SIR模型-Facebook.pdf', bbox_inches='tight',dpi=100)
plt.show()
