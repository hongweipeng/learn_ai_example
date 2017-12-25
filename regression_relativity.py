#coding: utf-8
import math
import numpy as np
from sklearn import linear_model

def computeCorrelation(x: list, y: list) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    SSR = 0
    var_x = 0  # x的方差
    var_y = 0  # y的方差
    for xi, yi in zip(x, y):
        diff_x = xi - x_mean
        diff_y = yi - y_mean
        SSR += diff_x * diff_y
        var_x += diff_x ** 2
        var_y += diff_y ** 2
    SST = math.sqrt(var_x * var_y)
    return  SSR / SST

def polyfit(x, y):
    linear = linear_model.LinearRegression()
    linear.fit(x, y)
    y_hat = linear.predict(x)
    y_mean = np.mean(y)
    SSR = 0
    SST = 0
    for i in range(len(y)):
        SSR += (y_hat[i] - y_mean) ** 2
        SST += (y[i] - y_mean) ** 2
    return SSR / SST

train_x = [1, 3, 8, 7, 9]
train_y = [10, 12, 24, 21, 34]

print(computeCorrelation(train_x, train_y))

train_x_2d = [[x] for x in train_x] # 通用的方式，训练集至少是二维的
print(polyfit(train_x_2d, train_y))

