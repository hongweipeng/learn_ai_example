# coding: utf-8
import tensorflow as tf

"""
placeholder
"""
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 运算，需要传入值给placeholder
    print(sess.run(output, feed_dict={
        input1: [7.0],
        input2: [2.0],
    }))




