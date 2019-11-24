import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
    # 定义一个指针

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # 线段树的一个特点就是叶子结点的数目 = 剩余节点数目（除了叶子结点）+1
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    # 添加树节点
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1   # 定位到第一个叶子结点的索引
        self.data[self.data_pointer] = data  # 将data数据放进去
        self.update(tree_idx, p)  # 更新一下树的父节点值，因为有可能发生了变化

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 当存满数据了之后重新存
            self.data_pointer = 0

    # 更新树节点值
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]   # 计算一下要更新的值与当前叶子节点的差值
        self.tree[tree_idx] = p            # 把更新的值填进对应的位置
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2   # 计算父节点位置的公式 = 子节点（idx） 除于二向下取整
            self.tree[tree_idx] += change    # 向上更新父节点的值

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the  methodin the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# Double DQN
class Priori_DDQNetwork:
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
            prioritized=True,
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
        self.prioritized = prioritized
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

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

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

        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        # 建立eval_net
        # 创建全连接层tf.layers.dense
        with tf.variable_scope('eval_net'):
            # first layer. collections is used later when assign to target net
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State

            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e1_1 = tf.layers.dense(e1, 10, tf.nn.sigmoid, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1_1')
            self.q_eval = tf.layers.dense(e1_1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

            # 计算损失单元
            with tf.variable_scope('loss'):
                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            # 训练网络单元
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # 建立target_net
        # 创建全连接层tf.layers.dense
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t1_1 = tf.layers.dense(t1, 10, tf.nn.sigmoid, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1_1')
            self.q_next = tf.layers.dense(t1_1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')


    # 存储记忆
    def store_transition(self, s, a, r, s_):
        # hasattr判断对象是否含有某种属性
        # 这里为了以防万一
        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:  # random replay
            # replace the old memory with new memory
            # 这个地方有一个技巧，memory_counter % self.memory_size
            # 事实上还是等于memory_counter,但是不会超过memory_size
            # 假设memory_size=100，那么index的范围就是0-99，当index=100时
            # 100 % 100 = 0，又会从头插入数据
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0

            transition = np.hstack((s, [a, r], s_))
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

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            # sample batch memory from all memory
            # 从记忆库中随机抽取部分数据（batch_memory）用于SGD更新参数
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
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

        if self.prioritized:
            _, abs_errors, cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target,
                                                                self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
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


