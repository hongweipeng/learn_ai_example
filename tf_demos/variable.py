# coding: utf-8
import tensorflow as tf
"""
在 tf 的结构中，只有通过 Variable 函数声明的，才能算是tf中的变量
"""
# 设置变量
state = tf.Variable(0, name='counter')

# 设置常量
one = tf.constant(1)

new_value = tf.add(state, one)        # 设置运算
update = tf.assign(state, new_value)  # 变量赋值

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


