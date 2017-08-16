import numpy as np
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32


import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gym_robot
import signal
import sys

EPISODES = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.actor = self._build_model()
        rospy.init_node('deepLearning', anonymous=False)
        rospy.Subscriber("obstacle", String, handleState)
        self.pub = rospy.Publisher('ev3Move', Float32, queue_size=1)  #0, 1, 2, 3

    def handleState(data):
        s = np.array(data)
        print(s)
        a = self.act(s)
        self.pub.publish(a)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
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
        actor.compile(optimizer='sgd', loss='mse')
        return actor


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action = self.actor.predict(state)
        assert action.shape == (self.action_size,)
        return action  # returns action


    def load(self, name):
        self.actor.load_weights(name)

    def save(self, name):
        self.actor.save_weights(name)


if __name__ == "__main__":    
    state_size = 2
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("./actor.h5")
    except IOError:
        pass
    done = False

    def soft_exit(signal, frame):
        print("Abort Training \n")
        var = raw_input("Want to save the model? y/n")
        if var == 'y':
            agent.save("./save/v3.h5")
        sys.exit(0)