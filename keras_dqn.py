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
from rl.callbacks import FileLogger,TestLogger

ENV_NAME = 'AutonomousRobot-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               target_model_update=1e-3, policy=policy)
             
dqn.compile(Adam(lr=1e-3), metrics=['mse'])
date = str(datetime.now())
log_filename = './logs/dqn_{}_{}_log.json'.format(ENV_NAME,date)
#log_filename = './logs/dqn_{}_test_log.json'.format(ENV_NAME)
callbacks = [FileLogger(log_filename, interval=25)]
csv_logger = CSVLogger('test.log')
#testCall = [TestLogger(FileLogger(log_filename, interval=25))]
#dqn.load_weights("dqn_AutonomousRobot-v0_2017-08-09 12:41:13.026280_weights.h5f")  
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2,callbacks=callbacks)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_{}_weights.h5f'.format(ENV_NAME,date), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5,verbose=2, visualize=True,callbacks=[csv_logger])
