import numpy as np
import gym
from datetime import datetime
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import gym_robot
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, TestLogger

# Only allocate memory as needed
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

ENV_NAME = 'AutonomousRobot-v3'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n

# model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Agent
memory = SequentialMemory(limit=2000000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-3, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mse'])

# logging
date = str(datetime.now())
log_filename = './logs/dqn_{}_{}_log.json'.format(ENV_NAME, date)
callbacks = [FileLogger(log_filename, interval=25)]
csv_logger = CSVLogger('test.log')

# load weights if needed
#dqn.load_weights("dqn_AutonomousRobot-v3_2017-08-17 18:43:23.361367_weights.h5f")

# Training
dqn.fit(env, nb_steps=2000000, visualize=False, verbose=2, callbacks=callbacks)

# save weights.
dqn.save_weights('dqn_{}_{}_weights.h5f'.format(
    ENV_NAME, date), overwrite=True)

# test
dqn.test(env, nb_episodes=5, verbose=2, visualize=True, callbacks=[csv_logger])
