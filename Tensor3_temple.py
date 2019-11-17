# 这是一个神经网络的练习
# 题目：https://www.cnblogs.com/babyfei/p/8085937.html

import tensorflow as tf
import numpy as np

# 添加网络的隐藏层
def add_layer(input_data, input_size, output_size, active_function = None):
    weights = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.zeros([1, output_size]) + 0.01)
    func = tf.matmul(input_data, weights) + bias
    if active_function == None:
        outputs = func
    else:
        outputs = active_function(func)
    return outputs

if __name__ == '__main__':
    # 数据准备
    data_set = np.array([[3.2, 9.6, 3.45, 2.15, 140, 2.8, 11.00, 50],
                         [3.2, 10.3, 3.75, 2.20, 120, 3.4, 10.90, 70],
                         [3.0, 9.0, 3.50, 2.20, 140, 3.5, 11.40, 50],
                         [3.2, 10.3, 3.65, 2.20, 150, 2.8, 10.80, 80],
                         [3.2, 10.1, 3.50, 2.00, 80, 1.5, 11.30, 50],
                         [3.4, 10.0, 3.40, 2.15, 130, 3.2, 11.50, 60],
                         [3.2, 9.6, 3.55, 2.10, 130, 3.5, 11.80, 65],
                         [3.0, 9.0, 3.50, 2.10, 100, 1.8, 11.30, 40],
                         [3.2, 9.6, 3.55, 2.10, 130, 3.5, 11.80, 65],
                         [3.2, 9.2, 3.50, 2.10, 140, 2.5, 11.00, 50],
                         [3.2, 9.5, 3.40, 2.15, 115, 2.8, 11.90, 50],
                         [3.9, 9.0, 3.10, 2.00, 80, 2.2, 13.00, 50],
                         [3.1, 9.5, 3.60, 2.10, 90, 2.7, 11.10, 70],
                         [3.2, 9.7, 3.45, 2.15, 130, 4.6, 10.85, 70]], dtype=np.float32
                        )

    y_data = np.array([2.24, 2.33, 2.24, 2.32, 2.2, 2.27, 2.2, 2.26,
                       2.2, 2.24, 2.24, 2.2, 2.2, 2.35], dtype=np.float32).reshape(14, 1)

    test_data = np.array([3.0, 9.3, 3.3, 2.05, 100, 2.8, 11.2, 50]).reshape(1, 8)

    xs = tf.placeholder(dtype=tf.float32, shape=[None, None])
    ys = tf.placeholder(dtype=tf.float32, shape=[None, None])

    # 创建网络结构
    layer1 = add_layer(xs, 8, 10, active_function=tf.sigmoid)
    layer2 = add_layer(layer1, 10, 4, active_function=tf.sigmoid)
    prediction = add_layer(layer2, 4, 1)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    operator = tf.train.GradientDescentOptimizer(0.35).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(10000):
            sess.run(operator, feed_dict={xs: data_set, ys: y_data})
            if _ % 50 == 0:
                print(sess.run(loss, feed_dict={xs: data_set, ys: y_data}))
            if _ == 9999:
                print(sess.run(prediction, feed_dict={xs: test_data}))










