# 设计思路是将神经网络分为神经元,网络层,以及整个网络三个部分,
# 首先定义sigmoid函数作为激活函数
import numpy as np


# 定义sigmoid函数以及sigmoid函数的导数logistic_derivative
def logistic(x):
    return 1/(1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))


# 神经元的设计
# BP神经网络可以看做是神经元的集合
# 神经元的主要功能
# 1.计算数据,输出结果
# 2.更新各连接权值
# 3.向上一层反馈权值更新值,实现反馈功能
# 4.每个神经元对于输入都有一个w权重,所有的输入共享这个神经元的共同的偏置
# 5.对于每个神经元权重的结构,这个是将每个神经元作为子类,所以w的结构即是(输入变量,1)

class Neuron:
    def __init__(self, len_input):

    self.weights = np.random.random(len_input)*0.1
    # 输入初始参数,随机取很小的值(<0.1)
    # len_input是输入到此神经元的数目
    self.input = np.ones(len_input)
    # 初始化当前实例的输入为1
    self.output = 1
    # 初始化当前神经元的输出为1
    self.deltas_item = 0
    # 误差项初始化为0
    self.last_weight_add = 0

    # 上一次权重增加的量,记录下来方便后面扩展时考虑增加冲量

    def calc_output(self, x):

    # 计算输出值
    self.input = x
    self.output = logistic(np.dot(self.weights.T, self.input))  # np.dot表示矩阵乘法
    return self.output

    def get_back_weight(self):
        re
