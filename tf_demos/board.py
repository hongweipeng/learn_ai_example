#coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

layer_num = 0
# 添加层
def add_layer(inputs, in_size, out_size, activation=None):
    global layer_num
    layer_num += 0
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]), name='W')
            tf.summary.histogram("layer%s-weights" % layer_num, weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram("layer%s-biases" % layer_num, biases)
        with tf.name_scope('y'):
            y = tf.matmul(inputs, weights) + biases
        if activation:
            outputs = activation(y)
        else:
            outputs = y
        tf.summary.histogram("layer%s-outputs" % layer_num, outputs)

        return outputs



# print(np.random.randn(10))

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

noise = np.random.normal(0.0, 0.05, x_data.shape)    # 添加噪点
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


layer1 = add_layer(xs, 1, 10, activation=tf.nn.relu)
layer2 = add_layer(layer1, 10, 1, activation=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1]), name='loss')
    tf.summary.scalar("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(0.1)

with tf.name_scope('train'):
    train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={
            xs: x_data,
            ys: y_data,
        })
        if i % 50 == 0:
            sess.run(loss, feed_dict={xs:x_data, ys: y_data})
            result = sess.run(merged, feed_dict={xs:x_data, ys: y_data})
            writer.add_summary(result, i)
# 在命令行运行 tensorboard --logdir=logs

