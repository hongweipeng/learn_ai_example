# coding: utf-8
import math
import random
import numpy as np
import collections
from sklearn import cluster, datasets


def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

class K_means(object):
    def __init__(self, k: int, max_iter=10):
        self.k = k
        self.max_iter = 10      # 最大迭代次数
        self.data_set = None    # 训练集
        self.labels = None      # 结果集

    def init_centroids(self) -> list:
        """
        从训练集中随机选择 k 个点作为质点
        """
        point_num = np.shape(self.data_set)[0]
        random_index = random.sample(list(range(point_num)), self.k)
        centroids = [self.data_set[i] for i in random_index]
        return centroids

    def fit(self, data_set):
        self.data_set = data_set
        point_num = np.shape(data_set)[0]
        self.labels = [ -1 ] * point_num            # 初始化结果集
        centroids = self.init_centroids()           # 初始化随机质点
        old_centroids = []                          # 上一次迭代的质点
        step = 0                                    # 当前迭代次数
        while not self.should_stop(old_centroids, centroids, step):
            old_centroids = np.copy(centroids)
            step += 1
            for i, point in enumerate(data_set):
                self.labels[i] = self.get_closest_index(point, centroids)
            centroids = self.update_centroids()

    def get_closest_index(self, point, centroids):
        min_dist = math.inf # 初始设为无穷大
        label = -1
        for i, centroid in enumerate(centroids):
            dist = euler_distance(centroid, point)
            if dist < min_dist:
                min_dist = dist
                label = i
        return label

    def update_centroids(self):
        """
        取各类的中心设为新的质点
        """
        collect = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            collect[label].append(self.data_set[i])

        centroids = []
        for i in range(self.k):
            centroids.append(np.mean(collect[i], axis=0))
        return centroids

    def should_stop(self, old_centroids, centroids, step) -> bool:
        """
        判断是否停止迭代，停止的条件是 新分类结果与原来一致或者已达到设置的迭代次数
        """
        if step > self.max_iter:
            return True
        return np.array_equal(old_centroids, centroids)


if __name__ == "__main__":
    iris = datasets.load_iris()

    my_k = K_means(4)
    my_k.fit(iris.data)
    print(np.array(my_k.labels))

    sk = cluster.KMeans(4)
    sk.fit(iris.data)
    print(sk.labels_)
