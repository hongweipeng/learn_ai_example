# coding: utf-8
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

class NeuralNetwork(object):
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
            #tmp = (np.random.random([layers[i], layers[i + 1]]) * 2 - 1) * 0.25
            tmp = (np.random.random([layers[i], layers[i + 1]]) * 2 - 1) * 0.25
            self.weights.append(tmp)

        # 偏向
        self.bias = []
        for i in range(1, len(layers)):
            self.bias.append((np.random.random(layers[i]) * 2 - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        y = np.array(y)
        # 随即梯度
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]   # 随即取某一条实例
            for j in range(len(self.weights)):
                a.append(self.activation(np.dot(a[j], self.weights[j]) + self.bias[j] ))
            errors = y[i] - a[-1]
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


if __name__ == "__main__":
    nn = NeuralNetwork(layers=[64, 100, 10])
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # 拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 分类结果离散化
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    nn.fit(X_train, labels_train)

    # 收集测试结果
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        predictions.append(np.argmax(o))

    # 打印对比结果
    print (confusion_matrix(y_test, predictions) )
    print (classification_report(y_test, predictions))