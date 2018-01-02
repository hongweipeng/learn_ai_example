#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def genData():
    train_x = []
    train_y = []
    with open("logistic_set.txt") as f:
        for line in f.readlines():
            line = line.strip().split()
            num = len(line)
            train_x.append([float(line[x]) for x in range(num - 1)])
            train_y.append(float(line[-1]))
    return train_x, train_y

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class LogisticReg(object):
    def __init__(self):
        pass
    def fit(self, x, y, learn_rate=0.0005):
        point_num, future_num = np.shape(x)
        new_x = np.ones(shape=(point_num, future_num + 1)) # 多一列x0，全部设为1
        new_x[:, 1:] = x
        self.theta = np.ones(shape=(future_num + 1, 1))

        x_mat = np.mat(new_x)
        y_mat = np.mat(y).T
        J = []
        for i in range(800):
            h = sigmoid(np.dot(x_mat, self.theta))
            # 打印损失函数
            cost = np.sum([ a * -np.log(b) + (1 - a) * -np.log(1 - b)  for a, b in zip(y_mat, h)])
            J.append(cost)
            self.theta -= learn_rate * x_mat.T * (h - y_mat)
        plt.plot(J)
        plt.show()

    def predict(self, row):
        row = np.array([1] + row)
        result = sigmoid(np.dot(row, self.theta))
        return 1 if result > 0.5 else 0

if __name__ == "__main__":
    mylog = LogisticReg()
    x, y = genData()
    test_row = [0.6, 12]
    mylog.fit(x, y)
    print(mylog.theta)
    print("LogisticReg predict:", mylog.predict(test_row))

    sk = LogisticRegression()
    sk.fit(x, y)
    print(sk.intercept_)
    print(sk.coef_)
    print("sklearn LogisticRegression predict:", sk.predict([test_row]))