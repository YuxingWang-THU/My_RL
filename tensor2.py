#使用tensorflow构建一个简单的三层神经网络

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#定义一个创建层（隐藏层）的函数
#一共有四部分组成：输入、输入大小、输出大小（神经元的数目）、激活函数
def add_layer(input, input_size, output_szie, activate_func=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_szie]))
    bias = tf.Variable(tf.zeros([1, output_szie])+0.1)
    func = tf.matmul(input, Weights) + bias
    if activate_func == None:
        outputs = func
    else:
        outputs = activate_func(func)
    return outputs


if __name__ == '__main__':
    x_data = np.linspace(-3, 3, 800, dtype=np.float32).reshape(800, 1)
    noise = np.random.normal(0, 0.001, x_data.shape)
    y_data = np.square(2 * np.cos(x_data)) + 0.5 * np.sin(x_data) + noise

    xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    layer1 = add_layer(xs, 1, 10, activate_func=tf.sigmoid)
    layer2 = add_layer(layer1, 10, 4, activate_func=tf.sigmoid)
    prediction = add_layer(layer2, 4, 1, activate_func=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    operator = tf.train.MomentumOptimizer(0.35, 0.1).minimize(loss)

    init = tf.global_variables_initializer()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_data, y_data)
    plt.ion()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(100000):
            sess.run(operator, feed_dict={xs: x_data, ys: y_data})
            if _ % 50 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                predication_value = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, predication_value, 'r-')
                plt.pause(0.1)

