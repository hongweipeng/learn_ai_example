# coding: utf-8
import numpy as np
from sklearn import linear_model

# 一元线性回归
class SimpleLinearRegression(object):
    """
    简单线性回归方程，即一元线性回归,只有一个自变量，估值函数为: y = b0 + b1 * x
    """
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
    def fit(self, x: list, y: list):
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        dinominator = 0
        numerator = 0
        for xi, yi in zip(x, y):
            numerator += (xi - x_mean) * (yi - y_mean)
            dinominator += (xi - x_mean) ** 2
        self.b1 = numerator / dinominator
        self.b0 = y_mean - self.b1 * x_mean

    def pridict(self, x):
        return self.b0 + self.b1 * x


class MyLinearRegression(object):
    def __init__(self):
        self.b = []

    def fit(self, x: list, y: list):
        # 为每条数据添加 1
        point_num, future_num = np.shape(x)
        tmpx = np.ones(shape=(point_num, future_num + 1))
        tmpx[:,1 :] = x
        x_mat = np.mat(tmpx)
        y_mat = np.mat(y).T
        xT = x_mat.T
        self.b = (xT * x_mat).I * xT * y_mat

    def predict(self, x):
        return np.mat([1] + x) * self.b

if __name__ == "__main__":
    x = [
        [100.0, 4.0],
        [50.0, 3.0],
        [100.0, 4.0],
        [100.0, 2.0],
        [50.0, 2.0],
        [80.0, 2.0],
        [75.0, 3.0],
        [65.0, 4.0],
        [90.0, 3.0],
        [90.0, 2.0]
    ]

    y = [9.3, 4.8, 8.9, 6.5, 4.2, 6.2, 7.4, 6.0, 7.6, 6.1]

    test_row = [50, 3]
    linear = MyLinearRegression()
    linear.fit(x, y)
    print(linear.predict(test_row)) # [[ 4.95830457]]

    sk = linear_model.LinearRegression()
    sk.fit(x, y)
    print(sk.predict([test_row])) # [ 4.95830457]




