#coding: utf-8
import numpy as np
import tensorflow as tf

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.3

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

# 损失函数
loss = tf.reduce_mean(tf.square(y - y_data))

# 创建优化器 使用梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5) # 参数表示学习率
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)   # 这句很容易被忽略
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))