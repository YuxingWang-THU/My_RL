import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# Double DQN
class DDQNetwork:
    def __init__(
            self,
            n_actions,                # 动作的数目
            n_features,               # 有几个state
            learning_rate=0.01,       # 学习率，系数阿尔法
            reward_decay=0.9,         # 奖励折扣，伽马
            e_greedy=0.9,             # e-greedy策略概率
            replace_target_iter=300,  # 更新Q-target的轮数
            memory_size=500,          # Memory库的大小
            batch_size=32,            # SGD所用的库中的条目数量
            e_greedy_increment=None,  # e的增量，随着学习的进行，为了保证算法的收敛，e应该逐渐增大
            output_graph=False,       # 是否输出计算图
            double_q=True,
            sess=None
    ):
        # 传递参数
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.1 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        # total learning step，用来记录学习进行到了哪一轮，为了后面更新Q-target做准备
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]，初始化memory，矩阵大小为（memory_size, n_features * 2 + 2）
        # n_features包括s,s_，所以要乘2，剩下的2对应a和r
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # 收集target_net和eval_net的参数
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        # 将eval_net的参数assign给target_net，完成更新
        with tf.variable_scope('hard_replacement'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 记录loss,用来画图
        self.cost_his = []

    # 建造DQN网络（target-net和eval-net两个网络）
    def _build_net(self):
        # 两个网络所需的所有参数，四元组
        # eval-net接收s,输出q-eval,现实值
        # target-net接收s_,输出q-next,目标值
        # Q(s, q-eval) += a * (r + gamma * maxQ(s_, q-next) - Q(s, q-eval))
        # 层设置
        # w_initializer:权重初始化，这里使用TensorFlow内置的函数生成正态分布
        # b_initializer:偏置初始化，这里初始化为常量0.1
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # 建立eval_net
        # 创建全连接层tf.layers.dense
        with tf.variable_scope('eval_net'):
            # first layer. collections is used later when assign to target net
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State

            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

            # 计算损失单元
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval, name='TD_error'))
            # 训练网络单元
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # 建立target_net
        # 创建全连接层tf.layers.dense
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')


    # 存储记忆
    def store_transition(self, s, a, r, s_):
        # hasattr判断对象是否含有某种属性
        # 这里为了以防万一
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 水平方向合并数组，方便存入memory
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        # 这个地方有一个技巧，memory_counter % self.memory_size
        # 事实上还是等于memory_counter,但是不会超过memory_size
        # 假设memory_size=100，那么index的范围就是0-99，当index=100时
        # 100 % 100 = 0，又会从头插入数据
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 动作选择
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    # 进行学习
    def learn(self):
        # check to replace target parameters
        # 当learn_step_counter = replace_target_iter的时候进行target-net参数更新
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('step_counter: %d \n target_params_replaced\n' % self.learn_step_counter)

        # sample batch memory from all memory
        # 从记忆库中随机抽取部分数据（batch_memory）用于SGD更新参数
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]


        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next,
                                     axis=1)  # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)  # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.s_: batch_memory[:, -self.n_features:],
                self.q_target: q_target
            })
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 画出损失函数
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)