import gym
from gym import spaces
from obstacles import Robot, Obstacle
import numpy as np
import random
try:
    from gym.envs.classic_control import rendering
    renderingAvailable = True
except Exception:
    renderingAvailable = False
import time
from random import randint




class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 1}

    def __init__(self):
        self.width = 600
        self.height = 600
        self.robot_width = 50
        self.robot_height = 30
        self.obstacles = []
        wall_size = 5
        x = randint(100, 500)
        y = randint(100,500)
        self.robot = Robot([x,y],40,25)
       # self.robot = Robot([self.width / 2, self.height / 2],
       #                    self.robot_width, self.robot_height)
        
        self.obstacle = Obstacle([500, 300], 50, 50)
        leftWall  = Obstacle([0,self.height/2],wall_size,self.height)
        rightWall = Obstacle([self.width,self.height/2],wall_size,self.height)
        topWall = Obstacle([self.width/2,self.height],self.width,wall_size)
        botWall = Obstacle([self.width/2,0],self.width,wall_size)
        self.obstacles = [self.obstacle,leftWall,rightWall,topWall,botWall]
        self.walls = [leftWall,rightWall,topWall,botWall]
        self.speed = 0.5
        self.pad_width = 1
        self.action_space = spaces.Discrete(3)  # Left, Right, Foward
        self.observation_space = spaces.Box(
            low=0, high=600, shape=(3,))

        self.viewer = None
        self.state = None

        self._reset()
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        
        if(action == 0):
            self.robot.move_forward()
        if(action == 1):
            self.robot.turn_left()
        if(action == 2):
            self.robot.turn_right()   
        reward, done = self.reward()
        if(action == 0):
            reward = reward + 2
        mins, p ,p = self.robot.usSensors(self.obstacles)
        self.state = np.array(mins)
        #print(self.state)
        return np.copy(self.state), reward, done, {}

    def reward(self):
        for obs in self.obstacles:         
            if self.robot.collision(obs):
                return -100, True
        return -1, False

    def _reset(self):
        self.state = np.zeros(3,)
        # x
        x = randint(100, 500)
        y = randint(100,500)
        a = randint(0,360)
        self.robot = Robot([x,y],40,25)
        self.robot.angle = a
        for obs in self.obstacles:         
            if self.robot.collision(obs):
               self._reset()
               break
        return np.copy(self.state)

    def _render(self, mode='human', close=False):
        if renderingAvailable:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return

            if self.viewer is None:
                self.viewer = rendering.Viewer(
                    self.width, self.height, display=self.display)

                robot = rendering.FilledPolygon(self.robot.get_drawing())
                c1 = rendering.make_circle(2)
                c2 = rendering.make_circle(2)
                c3 = rendering.make_circle(2)
                start = rendering.make_circle(3)
                obs = rendering.FilledPolygon(self.obstacle.get_drawing())
                for wall in self.walls:
                    draw = rendering.FilledPolygon(wall.get_drawing_static_position())
                    #draw.set_color(0,0,1)
                    self.viewer.add_geom(draw)
                self.obtrans = rendering.Transform()
                self.casttrans = rendering.Transform()
                self.casttrans2 = rendering.Transform()
                self.casttrans3 = rendering.Transform()
                self.starttrans = rendering.Transform()
                self.robottrans = rendering.Transform()
                robot.add_attr(self.robottrans)
                obs.add_attr(self.obtrans)
                c1.add_attr(self.casttrans)
                c2.add_attr(self.casttrans2)
                c3.add_attr(self.casttrans3)
                start.add_attr(self.starttrans)
                c1.set_color(1,0,0)
                c2.set_color(1,0,0)
                c3.set_color(1,0,0)
                start.set_color(1,0,0)
                self.viewer.add_geom(robot)
                self.viewer.add_geom(start)
                self.viewer.add_geom(obs)
                self.viewer.add_geom(c1)
                self.viewer.add_geom(c2)
                self.viewer.add_geom(c3)
                
            if self.state is None:
                return None

        #   min, points, pos = self.robot.rayCast(self.obstacles)
            min, points, pos = self.robot.usSensors(self.obstacles)
            #print(min)
            x, y = self.obstacle.get_postion()[0], self.obstacle.get_postion()[1]
            self.obtrans.set_translation(x, y)
            x = self.robot.get_postion()[0]
            y = self.robot.get_postion()[1]
            rot = self.robot.get_rotation()
            self.starttrans.set_translation(pos[0],pos[1])
            self.casttrans.set_translation(points[1][0],points[1][1])
            self.casttrans2.set_translation(points[0][0],points[0][1])
            self.casttrans3.set_translation(points[2][0],points[2][1])

            self.robottrans.set_translation(x, y)
            self.robottrans.set_rotation(rot * np.pi / 180)
            return self.viewer.render()
    
    def create_environment(self):
        robot = rendering.FilledPolygon(self.robot.get_drawing())
        cast = rendering.make_circle(2,)
        start = rendering.make_circle(3)
        obs = rendering.FilledPolygon(self.obstacle.get_drawing())
        self.obtrans = rendering.Transform()
        self.casttrans = rendering.Transform()
        self.starttrans = rendering.Transform()
        self.robottrans = rendering.Transform()