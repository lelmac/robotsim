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


class AutonomousRobotD(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 1}

    def __init__(self):
        self.width = 600
        self.height = 600
        self.robot_width = 50
        self.robot_height = 30
        self.obstacles = []
        self.target_position = [500, 100]
        wall_size = 5
        x = randint(100, 500)
        y = randint(100, 500)
        self.robot = Robot([x, y], 40, 25)
       # self.robot = Robot([self.width / 2, self.height / 2],
       #                    self.robot_width, self.robot_height)

        self.obstacle = Obstacle([500, 300], 50, 50)
        self.obstacle2 = Obstacle([100, 200], 35, 35)
        self.obstacle2.angle = 45
        leftWall = Obstacle([0, self.height / 2], wall_size, self.height)
        rightWall = Obstacle([self.width, self.height / 2],
                             wall_size, self.height)
        topWall = Obstacle([self.width / 2, self.height],
                           self.width, wall_size)
        botWall = Obstacle([self.width / 2, 0], self.width, wall_size)
        self.obstacles = [self.obstacle, self.obstacle2,
            leftWall, rightWall, topWall, botWall]
        self.walls = [leftWall, rightWall, topWall, botWall]
        self.speed = 0.5
        self.action_space = spaces.Discrete(3)  # Left, Right, Foward

        self.observation_space = spaces.Discrete(26 * 4 *3)
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

        infrared = self.robot.infraredSensor(self.obstacles)



        min, p, p = self.robot.singleUsSensors(self.obstacles)
        mins = np.abs(np.ceil((min-5)/10.0))
        reward, done = self.reward()

        self.state = [int(mins),infrared]
        if action == 0:
            reward += 2
        return self.state, reward, done, {}

    def reward(self, delta=0):
        #if(self.robot.pointInRobot(self.target_position)):
        #    return 800, True
        for obs in self.obstacles:
            if self.robot.collision(obs):
                return -700, True
        dis = np.linalg.norm(delta)
        #reward = -1 * np.e**(dis / 1000)
        reward = -1 
        return reward, False

    def _reset(self):
        self.state = [np.random.randint(0,25),np.random.randint(1,4)]
        # x
        #x = 200
        #y = 300
        #a = 0
        x = randint(100, 500)
        y = randint(100, 500)
        a = randint(0, 360)
        self.robot = Robot([x, y], 40, 25)
        self.robot.angle = a
        for obs in self.obstacles:
            if self.robot.collision(obs):
                self._reset()
                break
        return self.state

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
                target = rendering.make_circle(2)
                start = rendering.make_circle(3)
                obs = rendering.FilledPolygon(self.obstacle.get_drawing())
                obs2 = rendering.FilledPolygon(self.obstacle2.get_drawing())

                for wall in self.walls:
                    draw = rendering.FilledPolygon(
                        wall.get_drawing_static_position())
                    # draw.set_color(0,0,1)
                    self.viewer.add_geom(draw)
                self.obtrans = rendering.Transform()
                self.obtrans2 = rendering.Transform()
                self.casttrans = rendering.Transform()
                self.casttrans2 = rendering.Transform()
                self.casttrans3 = rendering.Transform()
                self.starttrans = rendering.Transform()
                self.robottrans = rendering.Transform()
                self.targettrans = rendering.Transform()
                robot.add_attr(self.robottrans)
                obs.add_attr(self.obtrans)
                obs2.add_attr(self.obtrans2)
                target.add_attr(self.targettrans)
                c1.add_attr(self.casttrans)
                c2.add_attr(self.casttrans2)
                c3.add_attr(self.casttrans3)
                start.add_attr(self.starttrans)
                c1.set_color(1, 0, 0)
                c2.set_color(1, 0, 0)
                c3.set_color(1, 0, 0)
                start.set_color(1, 0, 0)
                self.viewer.add_geom(robot)
                self.viewer.add_geom(start)
                self.viewer.add_geom(obs)
                self.viewer.add_geom(obs2)
                self.viewer.add_geom(c1)
                self.viewer.add_geom(c2)
                self.viewer.add_geom(c3)
                self.viewer.add_geom(target)

            if self.state is None:
                return None

            min, points, pos = self.robot.usSensors(self.obstacles)

            self.targettrans.set_translation(
                self.target_position[0], self.target_position[1])

            x, y = self.obstacle.get_postion(
            )[0], self.obstacle.get_postion()[1]
            self.obtrans.set_translation(x, y)
            x, y = self.obstacle2.get_postion(
            )[0], self.obstacle2.get_postion()[1]
            self.obtrans2.set_translation(x, y)
            self.obtrans2.set_rotation(self.obstacle2.angle * np.pi / 180)
            x = self.robot.get_postion()[0]
            y = self.robot.get_postion()[1]
            rot = self.robot.get_rotation()
            self.starttrans.set_translation(pos[0], pos[1])
            self.casttrans.set_translation(points[1][0], points[1][1])
            self.casttrans2.set_translation(points[0][0], points[0][1])
            self.casttrans3.set_translation(points[2][0], points[2][1])

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
