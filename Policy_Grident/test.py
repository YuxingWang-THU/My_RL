import gym
env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.action_space.sample())
print(env.observation_space.sample())
print(env.observation_space.high)
print(env.observation_space.low)



