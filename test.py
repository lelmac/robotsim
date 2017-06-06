import numpy as np
import gym

import gym_robot


ENV_NAME = 'robot-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.seed(123)
nb_actions = env.action_space.n
observation = env.reset()

for _ in range(400):
    env.render()
    #action = env.action_ssdpace.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(0)
for _ in range(90):
    env.render()
    #action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(1)
for _ in range(100):
    env.render()
    #action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(0)
for _ in range(180):
    env.render()
    #action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(2)  
for _ in range(500):
    env.render()
    #action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(0)
    if(done):
        env.reset()
