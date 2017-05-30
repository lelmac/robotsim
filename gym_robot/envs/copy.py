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
        self.width = 500
        self.height = 500
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
        self.move_pad(action)
        self.move_ball()
        reward, done = self.reward()
        return np.copy(self.board), reward, done, {}

    def reward(self):
        x = self.ball[0]
        y = self.ball[1]
        if y == self.height - 1:
            if self.pad_loc <= x < self.pad_loc + self.pad_width:
                self.place_ball()
                return 1, False 
            else:
                return -1, True
        else:
            return 0, False # The ball is still in the air

    # Moves the pad in the direction
    def move_pad(self, action):
        if action == STILL:
            return
        if action == LEFT:
            self.pad_loc = max(0, self.pad_loc - 1)
        if action == RIGHT:
            self.pad_loc = min(self.pad_loc + 1, self.width - self.pad_width)
        self.board[self.height-1] = 0
        self.board[self.height-1][self.pad_loc:self.pad_loc + self.pad_width] = 1

    # Moves the ball down by one
    def move_ball(self):
        x = self.ball[0]
        y = self.ball[1]
        self.board[y][x] = 0 
        self.board[y+1][x] = 1
        self.ball[1] += 1
    def move_robot(self):
        x = self.robot[0]
        y = self.robot[1]
    def move_forward(self):
        
    def _reset(self):
        self.board = np.zeros((self.height+1, self.width+1))
        self.board[self.height-1][:self.pad_width] = 1 # The pad
        self.pad_loc = 0
        self.place_ball()
        return np.copy(self.board)

    def place_ball(self):
        x = random.randint(0, self.width-1)
        self.board[0][x] = 1
        self.ball = [x,0]

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
        #if self.state is None: return None
        #x=self.state
        #robotx = x[0]+screen_width/2.0
        self.robottrans.set_translation(300, 300)
        time.sleep(0.4)

        return self.viewer.render()