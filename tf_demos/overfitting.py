# coding: utf-8
"""
过拟合问题, 使用 dropout 解决
"""
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

layer_num = 0
# 添加层
def add_layer(inputs, in_size, out_size, activation=None):
    global layer_num, keep_prob
    layer_num += 0
    weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    y = tf.matmul(inputs, weights) + biases
    y = tf.nn.dropout(y, keep_prob)   # 使用 dropout 防止过拟合
    if activation:
        outputs = activation(y)
    else:
        outputs = y
    tf.summary.histogram("layer%s-outputs" % layer_num, outputs)
    return outputs


digits = datasets.load_digits()
X = digits.data

y = digits.target
y = LabelBinarizer().fit_transform(y)

# 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 定义输入
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# 设计神经网络层
l1 = add_layer(xs, 64, 50, activation=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activation=tf.nn.softmax)

# 损失函数和优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    for i in range(501):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)




