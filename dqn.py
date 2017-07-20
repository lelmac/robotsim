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


EPISODES = 50000
SAVE_EP = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9994
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":    
    env = gym.make('AutonomousRobot-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("./save/v2.h5")
    except IOError:
        pass
    done = False

    def soft_exit(signal, frame):
        print("Abort Training \n")
        var = raw_input("Want to save the model? y/n")
        if var == 'y':
            agent.save("./save/v2.h5")
        sys.exit(0)
    signal.signal(signal.SIGINT, soft_exit)
    reward_history = []
    mvg_avg_history = []
    time_history = []
    batch_size = 64
    mv_avg = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        sum_reward = 0

        #print(str(e) + "/" + str(EPISODES))
        for time in range(1000):
            #if(e % 25 == 0):
            #    env.render()
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            
            sum_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                mv_avg = (mv_avg * (e) + sum_reward) / (e + 1)
                print("episode: {}/{}, score: {}, running average: {}, time: {} e: {:.2}"
                      .format(e, EPISODES, sum_reward, mv_avg, time, agent.epsilon))
                reward_history.append(sum_reward)
                mvg_avg_history.append(mv_avg)
                time_history.append(time)
                break
        if len(agent.memory) > batch_size:       
            agent.replay(batch_size)
        if(e % SAVE_EP == 0 and e != 0):
            name = "./save/v2" + str(e)  + ".h5" 
            agent.save(name)
            plt.plot(xrange(SAVE_EP),reward_history[e-SAVE_EP:e])
            plt.plot(xrange(SAVE_EP),mvg_avg_history[e-SAVE_EP:e])
            plt.legend(['Reward', 'Average Reward'], loc='upper left')
            plt.savefig("diagrams/" + str(e) + "reward.pdf")
            plt.clf()
            plt.plot(xrange(SAVE_EP),reward_history[e-SAVE_EP:e])
            plt.plot(xrange(SAVE_EP),time_history[e-SAVE_EP:e])
            plt.legend(['Reward', 'Time'], loc='upper left')
            plt.savefig("diagrams/" + str(e) + "time.pdf")
            plt.clf()
            mv_avg = 0
        
    #Endresultat
    plt.plot(xrange(EPISODES),reward_history)
    plt.plot(xrange(EPISODES),mvg_avg_history)
    plt.legend(['Reward', 'Average Reward'], loc='upper left')
    plt.savefig("diagrams/reward.pdf")
    plt.clf()
    plt.plot(xrange(EPISODES),reward_history)
    plt.plot(xrange(EPISODES),time_history)
    plt.legend(['Reward', 'Time'], loc='upper left')
    plt.savefig("diagrams/time.pdf")
