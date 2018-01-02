#coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 添加层
def add_layer(inputs, in_size, out_size, activation=None):
    weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    y = tf.matmul(inputs, weights) + biases
    if activation:
        outputs = activation(y)
    else:
        outputs = y
    return outputs



# print(np.random.randn(10))

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

noise = np.random.normal(0.0, 0.05, x_data.shape)    # 添加噪点
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


layer1 = add_layer(xs, 1, 10, activation=tf.nn.relu)
layer2 = add_layer(layer1, 10, 1, activation=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={
            xs: x_data,
            ys: y_data,
        })
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs:x_data, ys: y_data}))
