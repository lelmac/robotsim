#!/usr/bin/env python
import numpy as np
import rospy
import random
from std_msgs.msg import String
from std_msgs.msg import Int32

Q = np.zeros((26,4,3))
try:
    Q = np.load("qTable.npy")
except Exception:
    pass
print(Q)
lr = 0.4
discount = .95
num_episodes = 600
epsilon = 0.3
epsilon_decay = 0.95
curr_ep = 0
iterations_per_episode = []
rewards = []


def agent():
    global __pub__

    #initialize node publisher and subscriber
    rospy.init_node('deepLearning', anonymous=False)
    rospy.Subscriber("obstacle", String, handleState)
    __pub__ = rospy.Publisher('ev3Move', Int32, queue_size=1)  #0, 1, 2, 3

    #keep node running
    rospy.spin()



def handleState(data):
    
    print(data)
    s = np.array(data)
    s[1] = int(np.abs(np.ceil((s[1]-5)/10.0))) #discretize Input
    
    a = np.argmax(Q[s[0],s[1],:] + np.random.randn(1,312)*epsilon)
    __pub__.publish(a)
            




def calculateReward(lastState, currentState, lastAction, currentAction):
    pass


if __name__ == '__main__':
    agent()