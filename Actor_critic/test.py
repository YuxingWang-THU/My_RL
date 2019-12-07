import numpy as np
import tensorflow as tf
import gym
from brain import Actor, Critic

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time

LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make("CartPole-v0")
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)
sess.run(tf.initialize_all_variables())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    observation = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()
        action = actor.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        if done:
            reward = -20

        track_r.append(reward)

        td_error = critic.learn(observation, reward, observation_)
        actor.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        observation = observation_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

