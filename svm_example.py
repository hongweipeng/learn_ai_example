# coding: utf-8
import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape   # 获取图像数据集的形状，绘图使用

# 获取特征数据集和结果集
X = lfw_people.data
Y = lfw_people.target

n_features = X.shape[1]  # 特征的个数，或称为特征的维数
target_names = lfw_people.target_names # 数据集中有多少个人，以人名组成列表返回
n_classes = target_names.shape[0]
print("===== 数据集中信息 =====")
print("数据个数(n_samples):", n_samples)
print("特征个数，维度(n_features):", n_features)
print("结果集类别个数(n_classes):", n_classes)

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# 降维处理
n_components = 150
t0 = time.time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print("pca done %0.3fs" % (time.time() - t0))

# 从人脸中提取特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)
print("data set to pca done %0.3fs" % (time.time() - t0))

# 构造分类器
t0 = time.time()
param_grid = {
    "C": [1e3, 5e3, 1e4, 1e5],
    "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}

clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid)
clf.fit(X_train_pca, Y_train)
print("fit done %0.3fs" % (time.time() - t0))
print(clf.best_estimator_)

# 预测
t0 = time.time()
y_pred = clf.predict(X_test_pca)

print(classification_report(Y_test, y_pred, target_names=target_names))
print(confusion_matrix(Y_test, y_pred, labels=range(n_classes)))


# 测试结果可视化

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, Y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()