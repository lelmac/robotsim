import gym
from gym import spaces

import numpy as np
import random
from gym.envs.classic_control import rendering
import time

STILL, LEFT, RIGHT = 0, 1, 2

class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 1}

    def __init__(self):
        self.screen_width = 600
        self.screen_height = 600
        self.robot_width = 50
        self.robot_height = 30

        self.width = 500
        self.height = 500
        self.speed = 0.5
        self.pad_width = 1
        self.action_space = spaces.Discrete(3) # Left, Right, Foward
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width))

        self.viewer = None
        self.state = None

        self._reset()
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #self.move_pad(action)
        #self.move_ball()
        self.move_forward()
        reward, done = self.reward()
        return np.copy(self.state), reward, done, {}

    def reward(self):
        x = self.state[0]
        y = self.state[1]
        if y > self.screen_width-self.robot_width/2 or x < self.robot_width/2:
            return -1, True
        if x > self.screen_height-self.robot_height/2 or y < self.robot_height/2:
            return -1, True
        return 0, False # The ball is still in the air
    
    def move_forward(self):
        x = self.state[0]
        y = self.state[1]
        angle = self.state[2]
        x += np.sin(90 - angle**180 / np.pi)*self.speed
        y += np.sin(angle)*self.speed
        print(x)
        print(y)
        self.state[0] = x
        self.state[1] = y

    def _reset(self):
        self.state = np.zeros(5,)
        #x
        self.state[0] = 300
        #y
        self.state[1] = 300
        #rotation in degree
        self.state[2] = 0
        return np.copy(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 600
        robot_width = 30
        robot_height = 50
        
        

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            l,r,t,b = -robot_width/2,robot_width/2,robot_height/2,-robot_height/2
            robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            self.viewer.add_geom(robot)
        if self.state is None: return None

        x=self.state[0]
        y=self.state[1]
        #robotx = x[0]+screen_width/2.0
        self.robottrans.set_translation(x,y)

        return self.viewer.render()