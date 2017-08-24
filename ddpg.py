import numpy as np
import gym
from gym import wrappers
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
import gym_robot
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger

ENV_NAME = 'AutonomousRobot-v1'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#env = wrappers.Monitor(env, "./tmp/gym-results")
nb_actions = env.action_space.shape[0]
print(nb_actions)

# model with actor and critic
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(
    shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = merge([action_input, flattened_observation], mode='concat')
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(input=[action_input, observation_input], output=x)
print(critic.summary())

# create dddpg agent
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(
    size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.8, target_model_update=1e-3)
agent.compile(Adam(lr=.001), metrics=['mse'])


# logging
date = str(datetime.now())
log_filename = './logs/ddpg_{}_{}_log.json'.format(ENV_NAME, date)
callbacks = [FileLogger(log_filename, interval=25)]

# load weights if needed
# agent.load_weights('ddpg_{}_random_pos_weights.h5f'.format(ENV_NAME))

# Training
agent.fit(env, nb_steps=1000000, visualize=False, verbose=2,
          nb_max_episode_steps=1000, callbacks=callbacks)

# save the  weights.
agent.save_weights('ddpg_{}_random_pos_weights.h5f'.format(
    ENV_NAME), overwrite=True)

# test
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=1000)
