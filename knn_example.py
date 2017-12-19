# coding: utf-8
import math
import collections
import numpy as np

def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


class KNeighborsClass(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, data_set, labels):
        self.data_set = data_set
        self.labels = labels

    def predict(self, test_row):
        dist = []
        for v in self.data_set:
            dist.append(euler_distance(v, test_row))
        dist = np.array(dist)
        sorted_dist_index = np.argsort(dist) # 根据元素的值从大到小对元素进行排序，返回下标

        # 根据K值选出分类结果, ['A', 'B', 'B', 'A', ...]
        class_list = [ self.labels[ sorted_dist_index[i] ] for i in range(self.n_neighbors)]
        result_dict = collections.Counter(class_list)   # 计算各个分类出现的次数
        ret = sorted(result_dict.items(), key=lambda x: x[1], reverse=True) # 采用多数表决，即排序后的第一个分类
        return ret[0][0]

if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    knn = KNeighborsClass(n_neighbors=5)
    knn.fit(iris.data, iris.target)
    predict = knn.predict([0.1, 0.2, 0.3, 0.4])
    print(predict)  # output: 1