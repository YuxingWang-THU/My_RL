import gym
from RL_brain import Priori_DDQNetwork
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

sess = tf.Session()
RL = Priori_DDQNetwork(
        n_actions=3, n_features=2, memory_size=3000,
        e_greedy_increment=0.01, double_q=True, prioritized=True, sess=sess
    )
sess.run(tf.global_variables_initializer())
total_steps = 0


for i_episode in range(3):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # 车开得越高 reward 越大
        reward = abs(position - (-0.5)) * abs(position - (-0.5))

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()