#coding: utf-8
import random
import collections
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def logistic_deriv(x):
    """
    逻辑函数的导数
    """
    fx = logistic(x)
    return fx * (1 - fx)

class Network(object):
    def __init__(self, layers: list):
        self.num_layers = len(layers)       # 神经网络层数
        self.activation = logistic
        self.activation_deriv = logistic_deriv
        # 初始化随机权重
        self.weights = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))

        # 偏向
        self.bias = []
        for i in range(1, self.num_layers):
            self.bias.append(np.random.randn(layers[i]))

    def feedforward(self, a):
        for w, b in zip(self.weights, self.bias):
            a = self.activation(np.dot(a, w) + b)
        return a

    def SGD(self, train_data: list, epochs: int, mini_batch_size=100, eta=0.5, test_data=None):
        n = len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            mini_batches = [ train_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), len(test_data)))

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.bias, nabla_b)]


    def backprop(self, x, y):
        # 正向传播
        activations = [x, ]
        #zs = [] # 未经过激活函数的输出向量
        for i in range(len(self.weights)):
            z = np.dot(activations[i], self.weights[i]) + self.bias[i]
            #zs.append(z)
            activations.append(self.activation(z))

        # 反向传播
        delta = self.cost_derivative(activations[-1], y) * self.activation_deriv(activations[-1])
        deltas = [delta, ]
        for i in range(len(activations) - 2, 0, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * self.activation_deriv(activations[i]))
            deltas.reverse()
        weights = []
        bias = []
        for i in range(len(self.weights)):
            bias.append(deltas[i])
            layer = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            weights.append(np.dot(layer.T, delta))
        return bias, weights

    def cost_derivative(self, output, y):
        return output - y
    def evaluate(self, test_data):
        predictions = []
        for x, y in test_data:
            o = self.feedforward(x)
            predictions.append(np.argmax(o) == np.argmax(y))
        counter = collections.Counter(predictions)
        return counter[True]

if __name__ == "__main__":
    nn = Network(layers=[784, 30, 10])
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
    train_data = random.sample(train_data, 50000)
    nn.SGD(train_data, 30, mini_batch_size=30, eta=3.0, test_data=test_data)
