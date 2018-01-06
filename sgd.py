#coding: utf-8
"""
随即梯度下降算法
"""
import random
import collections
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    """
    tanh的导数
    """
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def logistic_deriv(x):
    """
    逻辑函数的导数
    """
    fx = logistic(x)
    return fx * (1 - fx)

class NeuralNetworkSGD(object):
    def __init__(self, layers, activation='logistic'):
        """
        :param layers: 层数，如[4, 3, 2] 表示两层len(list)-1,(因为第一层是输入层，，有4个单元)，
        第一层有3个单元，第二层有2个单元
        :param activation:
        """
        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv

        # 初始化随即权重
        self.weights = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))

        # 偏向
        self.bias = []
        for i in range(1, len(layers)):
            self.bias.append(np.random.randn(layers[i]))

    def update_batch(self, batch, learning_rate=0.2):
        # 随即梯度
        for x, y in batch:
            #i = np.random.randint(X.shape[0])
            a = [x, ]   # 随即取某一条实例
            for j in range(len(self.weights)):
                a.append(self.activation(np.dot(a[j], self.weights[j]) + self.bias[j] ))
            errors = y - a[-1]
            deltas = [errors * self.activation_deriv(a[-1]) ,]  # 输出层的误差
            # 反向传播，对于隐藏层的误差
            for j in range(len(a) - 2, 0, -1):
                tmp = np.dot(deltas[-1], self.weights[j].T) * self.activation_deriv(a[j])
                deltas.append(tmp)
            deltas.reverse()

            # 更新权重
            for j in range(len(self.weights)):
                layer = np.atleast_2d(a[j])
                delta = np.atleast_2d(deltas[j])
                self.weights[j] += learning_rate * np.dot(layer.T, delta)

            # 更新偏向
            for j in range(len(self.bias)):
                self.bias[j] += learning_rate * deltas[j]

    def predict(self, row):
        a = np.array(row) # 确保是 ndarray 对象
        for i in range(len(self.weights)):
            a = self.activation(np.dot(a, self.weights[i]) + self.bias[i])
        return a

    def feedforward(self, a):
        for w, b in zip(self.weights, self.bias):
            a = self.activation(np.dot(a, w) + b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """

        :param train_data:
        :param epochs:
        :param mini_batch_size:
        :param eta: 学习率
        :param test_data:
        :return:
        """
        n_test = 0
        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        for i in range(epochs):
            mini_batch = random.sample(train_data, mini_batch_size)
            self.update_batch(mini_batch, eta)
            if i % 50 == 0 and n_test:
                print("Epoch {0}: {1} / {2}".format( i, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        predictions = []
        for x, y in test_data:
            o = self.predict(x)
            predictions.append(np.argmax(o) == np.argmax(y))
        counter = collections.Counter(predictions)
        return counter[True]

if __name__ == "__main__":
    nn = NeuralNetworkSGD(layers=[784, 100, 10])
    digits = datasets.fetch_mldata('mnist-original')
    X = digits.data
    y = digits.target
    y = LabelBinarizer().fit_transform(y)# 分类结果离散化
    # 拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60000)

    # 分类结果离散化
    #labels_train = LabelBinarizer().fit_transform(y_train)
    #labels_test = LabelBinarizer().fit_transform(y_test)

    train_data = [(a, b) for a, b in zip(X_train, y_train)]
    test_data = [(a, b) for a, b in zip(X_test, y_test)]
    nn.SGD(train_data, 1000, mini_batch_size=100, eta=0.25, test_data=test_data)