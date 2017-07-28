import gym
import numpy as np
import gym_robot
from matplotlib import pyplot as plt

env = gym.make('AutonomousRobot-v2')

Q = np.zeros([env.observation_space.n,env.action_space.n])

lr = .8
discount = .95
num_episodes = 100


iterations_per_episode = []
rewards = []
for episode in range(num_episodes):
    #Reset environment
    s = env.reset()
    rAll = 0
    d = False
    iteration = 0

    while iteration < 1000:
        if episode % 25 == 0:
            env.render()
        
        iteration+=1
        #Choose an action by greedily
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))
        #Get new state and reward from environment
        s1,r,done,_ = env.step(a)

        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + discount*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        #update State
        s = s1
        if done == True:
            print("episode: {}/{}, score: {}, time: {} "
                      .format(episode, num_episodes, rAll,iteration))
            break

    rewards.append(rAll)
    iterations_per_episode.append(iteration)
print "Score over time: " +  str(sum(rewards)/num_episodes)

print "Final Q-Table Values"
print Q

#Plot that stuff
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot(range(num_episodes),rewards)
axarr[0].set_ylabel("Reward")
axarr[1].plot(range(num_episodes),iterations_per_episode)
axarr[1].set_ylabel("Iterationen")
plt.xlabel('Episoden')
plt.savefig("diagrams/rewardQLearning.pdf")
plt.show()