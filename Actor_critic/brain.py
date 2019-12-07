import numpy as np
import tensorflow as tf

GAMMA = 0.9     # reward discount in TD error

# 其实，我只是个演员
# 每次入账一条observation，矩阵大小[1, n_features]
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        # 类属性定义，使用placeholder
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        # 网络参数的定义，两层dense
        # 和policy gradient一样
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            #log_prob = tf.log(self.acts_prob[0, self.a])
            neg_log_prob = tf.reduce_sum(tf.log(self.acts_prob)*tf.one_hot(self.a, n_actions), axis=1)
            self.exp_v = tf.reduce_mean(neg_log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


# 其实，我只是一个喷子
# 每次入账一条observation，矩阵大小[1, n_features]
# 是为了做到及时评价，这点和PG不太一样，PG是回合学习
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):

        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        # 建造网络
        # 输入：observation，大小为：[1, n_features]
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            # 输出的是V(S)，注意
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        # 计算TD_error,这里使用的是时间差分
        with tf.variable_scope('squared_TD_error'):
            # V_现实：self.r + GAMMA * self.v_
            # V_估计：self.v
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    # 学习，Critic的学习率要大一些
    # policy gradient 对应的是：某状态下采取的动作的概率 * 动作的奖励（基于策略函数）
    # Actor Critic 对应的是：某状态下采取的动作的概率 * 该状态的价值（基于值函数 * 基于策略函数）
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


