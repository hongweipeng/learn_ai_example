#coding: utf-8
import collections
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
"""
分类问题
"""
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

def compute_accuracy(v_xs, v_ys, sess: tf.Session):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    result = sess.run(correct_prediction) # [False False False True False]
    counter = collections.Counter(result)
    return counter[True] /  len(result)
    #accuarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #result = sess.run(accuarcy, feed_dict={xs: v_xs, ys: v_ys})
    #return result


# 数据集
digist = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义输入
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

def no_cnn():
    # 神经网络层
    prediction = add_layer(xs, 784, 10, activation=tf.nn.softmax)

    # 损失函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # 随机梯度
        for i in range(501):
            batch_xs, batch_ys = digist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            if i % 50 == 0:
                print(compute_accuracy(batch_xs, batch_ys, sess))


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, weights):
    """
    设置卷积神经网络
    strides [1, x_movement, y_movement, 1]
    strides[0] 必须与 strides[3] 相等
    """
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')